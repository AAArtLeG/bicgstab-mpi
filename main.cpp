#include <mpi.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <iomanip>

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <limits>
#include <random>

#define GM 398600.5
#define R 6378000
#define PI 3.14159265358979323846  
#define N 902
#define D 500000

using namespace std;

double scalarProduct(const vector<double>& vec1, const vector<double>& vec2) {
	if (vec1.size() != vec2.size()) {
		throw runtime_error("scalarProduct: vector sizes do not match");
	}

	double sum = 0.0;
	for (size_t i = 0; i < vec1.size(); i++) {
		sum += vec1[i] * vec2[i];
	}
	return sum;
}

double scalarProductLocal(const vector<double>& a, const vector<double>& b, int istart, int iend) {
	double sum = 0.0;
	for (int i = istart; i <= iend; i++) {
		sum += a[i] * b[i];
	}
	return sum;
}

double vectorNorm(const vector<double>& vec) {
	return sqrt(scalarProduct(vec, vec));
}

vector<double> vectorAdd(const vector<double>& a, const vector<double>& b) {
	size_t size = a.size();
	vector<double> result(size);
	for (size_t i = 0; i < size; i++) {
		result[i] = a[i] + b[i];
	}
	return result;
}

vector<double> vectorSubtract(const vector<double>& a, const vector<double>& b) {
	size_t size = a.size();
	vector<double> result(size);
	for (size_t i = 0; i < size; i++) {
		result[i] = a[i] - b[i];
	}
	return result;
}

vector<double> vectorScale(const vector<double>& vec, double factor) {
	size_t size = vec.size();
	vector<double> result(size);
	for (size_t i = 0; i < size; i++) {
		result[i] = vec[i] * factor;
	}
	return result;
}

vector<double> matrixVectorMultiply(const vector<vector<double>>& mat, const vector<double>& vec) {
	size_t n = mat.size();
	vector<double> prod(n, 0.0);
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < mat[i].size(); j++) {
			prod[i] += mat[i][j] * vec[j];
		}
	}
	return prod;
}

void matrixVectorMultiplyLocal(
	const vector<double>& A_local,
	int rowsLocal,
	int n,
	const vector<double>& x,
	vector<double>& y_local)
{
	y_local.assign(rowsLocal, 0.0);

	for (int iloc = 0; iloc < rowsLocal; iloc++) {
		double sum = 0.0;
		for (int j = 0; j < n; j++) {
			sum += A_local[iloc * n + j] * x[j];
		}
		y_local[iloc] = sum;
	}
}

vector<double> solveWithBiCGSTAB(
	const vector<vector<double>>& mat,
	const vector<double>& b,
	vector<double>& residualHistory,
	double tol = 1e-10,
	int maxIterations = 1000,
	MPI_Comm comm = MPI_COMM_WORLD)
{
	/*int nprocs, myrank;
	MPI_Comm_size(comm, &nprocs);
	MPI_Comm_rank(comm, &myrank);*/

	size_t n = mat.size();

	vector<double> x(n, 0.0);
	vector<double> r = vectorSubtract(b, matrixVectorMultiply(mat, x));
	vector<double> r_star = r;
	vector<double> p = r;
	vector<double> s;

	double alpha, omega, beta;
	vector<double> xNew, rNew, pNew;

	residualHistory.push_back(vectorNorm(r));

	// newest: matrixVectorMultiply(mat, p) toklo eta paralizovat i sobrat v glob alphu

	for (int j = 0; j < maxIterations; ++j) {
		//alpha ets maje buti loc
		alpha = scalarProduct(r, r_star) / scalarProduct(r_star, matrixVectorMultiply(mat, p));

		//s ets maje buti loc
		s = vectorSubtract(r, vectorScale(matrixVectorMultiply(mat, p), alpha));
		//omega eta maje buti loc
		omega = scalarProduct(matrixVectorMultiply(mat, s), s) /
			scalarProduct(matrixVectorMultiply(mat, s), matrixVectorMultiply(mat, s));

		// a eto uzhe vse iz omega, s, alpha kotorije uze cerez MPI_Allgatherv sobrani v full versiu
		xNew = vectorAdd(vectorAdd(x, vectorScale(p, alpha)), vectorScale(s, omega));

		rNew = vectorSubtract(s, vectorScale(matrixVectorMultiply(mat, s), omega));

		beta = scalarProduct(rNew, r_star) / scalarProduct(r, r_star) * alpha / omega;

		pNew = vectorAdd(
			rNew,
			vectorScale(
				vectorSubtract(p, vectorScale(matrixVectorMultiply(mat, p), omega)),
				beta
			)
		);

		x = xNew;
		r = rNew;
		p = pNew;

		double r_norm = vectorNorm(r);
		residualHistory.push_back(r_norm);

		if (r_norm < tol) {
			break;
		}
	}

	return x;
}

vector<double> solveWithBiCGSTAB_MPI(
	const vector<double>& A_local,
	const vector<double>& b,
	vector<double>& residualHistory,
	double tol = 1e-10,
	int maxIterations = 1000,
	int rowsLocal = 0,
	int istart = 0,
	int iend = 0,
	int* recvcounts = nullptr,
	int* displs = nullptr,
	MPI_Comm comm = MPI_COMM_WORLD)
{
	int nprocs, myrank;
	MPI_Comm_size(comm, &nprocs);
	MPI_Comm_rank(comm, &myrank);

	int n = b.size();

	vector<double> x(n, 0.0);
	vector<double> r = b;
	vector<double> r_star = r;
	vector<double> p = r;
	vector<double> s(n, 0.0);

	double alpha, omega, beta;
	vector<double> xNew(n, 0.0), rNew(n, 0.0), pNew(n, 0.0);

	vector<double> v(n, 0.0), t(n, 0.0);
	vector<double> v_local(rowsLocal, 0.0), t_local(rowsLocal, 0.0);

	double rr_local = scalarProductLocal(r, r, istart, iend);
	double rr_global = 0.0;
	MPI_Allreduce(&rr_local, &rr_global, 1, MPI_DOUBLE, MPI_SUM, comm);
	residualHistory.push_back(sqrt(rr_global));

	for (int j = 0; j < maxIterations; ++j) {
		//if (myrank == 0 && j % 10 == 0) cout << "iter = " << j << endl;

		matrixVectorMultiplyLocal(A_local, rowsLocal, n, p, v_local);

		MPI_Allgatherv(&v_local[0], rowsLocal, MPI_DOUBLE, &v[0], recvcounts, displs, MPI_DOUBLE, comm);

		double alpha_citat_local = scalarProductLocal(r, r_star, istart, iend);
		double alpha_menov_local = scalarProductLocal(r_star, v, istart, iend);

		double alpha_citat_global = 0.0;
		double alpha_menov_global = 0.0;

		MPI_Allreduce(&alpha_citat_local, &alpha_citat_global, 1, MPI_DOUBLE, MPI_SUM, comm);
		MPI_Allreduce(&alpha_menov_local, &alpha_menov_global, 1, MPI_DOUBLE, MPI_SUM, comm);

		alpha = alpha_citat_global / alpha_menov_global;

		s = vectorSubtract(r, vectorScale(v, alpha));

		matrixVectorMultiplyLocal(A_local, rowsLocal, n, s, t_local);

		MPI_Allgatherv( &t_local[0], rowsLocal, MPI_DOUBLE, &t[0], recvcounts, displs, MPI_DOUBLE, comm);

		double omega_citat_local = scalarProductLocal(t, s, istart, iend);
		double omega_menov_local = scalarProductLocal(t, t, istart, iend);

		double omega_citat_global = 0.0;
		double omega_menov_global = 0.0;

		MPI_Allreduce(&omega_citat_local, &omega_citat_global, 1, MPI_DOUBLE, MPI_SUM, comm);
		MPI_Allreduce(&omega_menov_local, &omega_menov_global, 1, MPI_DOUBLE, MPI_SUM, comm);

		omega = omega_citat_global / omega_menov_global;

		xNew = vectorAdd(vectorAdd(x, vectorScale(p, alpha)), vectorScale(s, omega));
		rNew = vectorSubtract(s, vectorScale(t, omega));

		double beta_citat_local = scalarProductLocal(rNew, r_star, istart, iend);
		double beta_menov_local = scalarProductLocal(r, r_star, istart, iend);

		double beta_citat_global = 0.0;
		double beta_menov_global = 0.0;

		MPI_Allreduce(&beta_citat_local, &beta_citat_global, 1, MPI_DOUBLE, MPI_SUM, comm);
		MPI_Allreduce(&beta_menov_local, &beta_menov_global, 1, MPI_DOUBLE, MPI_SUM, comm);

		beta = beta_citat_global / beta_menov_global * alpha / omega;

		pNew = vectorAdd(
			rNew,
			vectorScale(
				vectorSubtract(p, vectorScale(v, omega)),
				beta
			)
		);

		x = xNew;
		r = rNew;
		p = pNew;

		double r_norm_local = scalarProductLocal(r, r, istart, iend);
		double r_norm_global = 0.0;
		MPI_Allreduce(&r_norm_local, &r_norm_global, 1, MPI_DOUBLE, MPI_SUM, comm);

		double r_norm = sqrt(r_norm_global);
		residualHistory.push_back(r_norm);

		if (r_norm < tol) {
			break;
		}
	}

	return x;
}


int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);

	int myrank, nprocs;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	int nlocal = N / nprocs;
	if (N % nprocs != 0) nlocal++;

	int nlast = N - (nprocs - 1) * nlocal;

	int istart = myrank * nlocal;
	int iend = istart + nlocal - 1;

	if (myrank == nprocs - 1) {
		iend = N - 1;
	}

	int* recvcounts = new int[nprocs];
	int* displs = new int[nprocs]; //otkial v reciver berieme hodnoty pre process

	for (int p = 0; p < nprocs - 1; p++) {
		recvcounts[p] = nlocal;
		displs[p] = p * nlocal;
	}

	recvcounts[nprocs - 1] = nlast;
	displs[nprocs - 1] = (nprocs - 1) * nlocal;

	vector<double> Xx(N), Xy(N), Xz(N);
	vector<double> nx(N), ny(N), nz(N);
	vector<double> Sx(N), Sy(N), Sz(N);
	vector<double> qTest(N);
	vector<double> q(N, 0.0);
	vector<double> B(N), L(N), H(N);

	if (myrank == 0) {
		ifstream file;
		file.open("BL-902.dat", ios::in);

		if (!file.is_open()) {
			cout << "Subor sa nepodarilo otvorit.\n";
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		/*vector<double> B;
		vector<double> L;
		vector<double> H;*/
		vector<double> S;
		

		double b, l, h, qq, skip;

		int idx = 0;
		while (file >> b >> l >> h >> qq >> skip) {
			if (idx >= N) {
				cout << "Error: too many rows in file\n";
				MPI_Abort(MPI_COMM_WORLD, 1);
			}

			B[idx] = b;
			L[idx] = l;
			H[idx] = h;
			q[idx] = qq;
			idx++;
		}

		file.close();

		if (B.size() != N) {
			cout << "Error: file has " << B.size() << " rows, but N = " << N << endl;
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		cout << "Pocet nacitanych riadkov: " << B.size() << endl;

		for (size_t i = 0; i < B.size(); i++) {
			//cout << B[i] << " " << L[i] << " " << H[i] << " " << q[i] << endl;
		}

		/*for (size_t i = 0; i < B.size(); i++) {
			H[i] = 0.0;
		}*/

		for (size_t i = 0; i < q.size(); i++) {
			q[i] *= -0.00001;
		}

		for (int i = 0; i < N; i++) {
			double Brad = B[i] * PI / 180.0;
			double Lrad = L[i] * PI / 180.0;

			Xx[i] = (R + H[i]) * cos(Brad) * cos(Lrad);
			Xy[i] = (R + H[i]) * cos(Brad) * sin(Lrad);
			Xz[i] = (R + H[i]) * sin(Brad);

			nx[i] = cos(Brad) * cos(Lrad);
			ny[i] = cos(Brad) * sin(Lrad);
			nz[i] = sin(Brad);
		}

		for (int i = 0; i < N; i++) {
			double Brad = B[i] * PI / 180.0;
			double Lrad = L[i] * PI / 180.0;

			Sx[i] = (R - D + H[i]) * cos(Brad) * cos(Lrad);
			Sy[i] = (R - D + H[i]) * cos(Brad) * sin(Lrad);
			Sz[i] = (R - D + H[i]) * sin(Brad);
		}

		for (int i = 0; i < N; i++) {
			//qTest[i] = -GM / (R * R);
		}
	}

	MPI_Bcast(Xx.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(Xy.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(Xz.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	MPI_Bcast(nx.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(ny.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(nz.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	MPI_Bcast(Sx.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(Sy.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(Sz.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//MPI_Bcast(qTest.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(q.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	int rowsLocal = iend - istart + 1;
	vector<double> GN_local(rowsLocal * N);

	cout << "rank = " << myrank
		<< " istart = " << istart
		<< " iend = " << iend
		<< " rowsLocal = " << rowsLocal << endl;

	for (int i = istart; i <= iend; i++) {
		int iloc = i - istart;

		for (int j = 0; j < N; j++) {
			double dx = Xx[i] - Sx[j];
			double dy = Xy[i] - Sy[j];
			double dz = Xz[i] - Sz[j];

			double tempScalSuc = dx * nx[i] + dy * ny[i] + dz * nz[i];
			double rij = sqrt(dx * dx + dy * dy + dz * dz);

			GN_local[iloc * N + j] = -tempScalSuc / (4.0 * PI * rij * rij * rij);
		}
	}

	//vector<vector<double>> GN(N, vector<double>(N));

	////zaminit na to chto scitaju v zaleznosti vid ranku svoj kusok GN
	//double tempScalSuc;
	//double rij;
	//for (int i = 0; i < N; i++) {
	//	for (int j = 0; j < N; j++) {
	//		tempScalSuc = (X[0][i] - S[0][j]) * n[0][i] + (X[1][i] - S[1][j]) * n[1][i] + (X[2][i] - S[2][j]) * n[2][i];
	//		rij = sqrt((X[0][i] - S[0][j]) * (X[0][i] - S[0][j]) + (X[1][i] - S[1][j]) * (X[1][i] - S[1][j]) + (X[2][i] - S[2][j]) * (X[2][i] - S[2][j]));
	//		GN[i][j] = -tempScalSuc / (4.0 * PI * rij * rij * rij);
	//	}
	//}

	//ofstream filee;
	//for (int i = 0; i < 5; i++) {
	//	for (int j = 0; j < 30; j++) {
	//		cout << GN[i][j] << " ";
	//	}
	//	cout << endl;
	//}

	//filee.open("GN.txt", ios::out);

	//if (!filee.is_open()) {
	//	cout << "Subor sa nepodarilo otvorit.\n";
	//	return 1;
	//}

	//for (int i = 0; i < N; i++) {
	//	for (int j = 0; j < N; j++) {
	//		filee << GN[i][j] << " ";
	//	}
	//	filee << endl;
	//}

	//filee.close();

	cout << "rank " << myrank << " before solver" << endl;

	vector<double> residHistoryBiCGSTAB;
	vector<double> alphaTest = solveWithBiCGSTAB_MPI(GN_local, q, residHistoryBiCGSTAB, 1e-7, 200, rowsLocal, istart, iend, recvcounts, displs, MPI_COMM_WORLD);

	cout << "rank " << myrank << " after solver" << endl;

	if(myrank == 0){
		cout << N << endl;

		cout << "Solution BiCGSTAB method:\n";
		for (size_t i = 0; i < N; i++) {
			//cout << "alpha[" << i << "] = " << alphaTest[i] << "\n";
		}

		for (size_t i = 0; i < residHistoryBiCGSTAB.size(); i++) {
			cout << "residHistoryBiCGSTAB[" << i << "] = " << residHistoryBiCGSTAB[i] << "\n";
		}
	}

	/*vector<vector<double>> G(N, vector<double>(N));

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			rij = sqrt((X[0][i] - S[0][j]) * (X[0][i] - S[0][j]) + (X[1][i] - S[1][j]) * (X[1][i] - S[1][j]) + (X[2][i] - S[2][j]) * (X[2][i] - S[2][j]));
			G[i][j] = 1 / (4 * PI * rij);
		}
	}

	vector<double> u = matrixVectorMultiply(G, alphaTest);

	cout << "Solution u:\n";
	for (size_t i = 0; i < N; i++) {
		cout << "u[" << i << "] = " << u[i] << "\n";
	}*/

	vector<double> u_local(rowsLocal, 0.0);

	for (int i = istart; i <= iend; i++) {
		int iloc = i - istart;
		double sum = 0.0;

		for (int j = 0; j < N; j++) {
			double dx = Xx[i] - Sx[j];
			double dy = Xy[i] - Sy[j];
			double dz = Xz[i] - Sz[j];

			double rij = sqrt(dx * dx + dy * dy + dz * dz);
			double gij = 1.0 / (4.0 * PI * rij);

			sum += gij * alphaTest[j];
		}

		u_local[iloc] = sum;
	}

	vector<double> u;

	if (myrank == 0) {
		u.resize(N);
	}

	MPI_Gatherv(&u_local[0], rowsLocal, MPI_DOUBLE, myrank == 0 ? &u[0] : nullptr, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (myrank == 0) {
		cout << "Solution u:\n";
		for (int i = 0; i < N; i++) {
			cout << "u[" << i << "] = " << u[i] << "\n";
		}
	}

	if (myrank == 0) {
		ofstream fout("solution_u.dat");

		if (!fout.is_open()) {
			cerr << "Cannot create solution_u.dat" << endl;
		}
		else {
			fout << fixed;

			for (int i = 0; i < N; i++) {
				fout << setw(10) << setprecision(2) << B[i] << " "
					<< setw(11) << setprecision(6) << L[i] << " "
					<< scientific << setprecision(3) << u[i] << "\n";
			}

			fout.close();
			cout << "Solution saved to solution_x.dat" << endl;
		}
	}

	MPI_Finalize();

    return 0;
}
