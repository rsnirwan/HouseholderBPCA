functions{
    matrix V_low_tri_plus_diag_unnormed (int D, int Q, vector v) {
        // Put parameters into lower triangular matrix
        matrix[D, Q] V;

        int idx = 1;
        for (d in 1:D) {
            for (q in 1:Q) {
                if (d >= q) {
                    V[d, q] = v[idx];
                    idx += 1;
                } else
                V[d, q] = 0;
            }
        }
        return V;
    }
    matrix V_low_tri_plus_diag (int D, int Q, vector v) {
        matrix[D, Q] V = V_low_tri_plus_diag_unnormed(D, Q, v);
        for (q in 1:Q){
            V[,q] = V[,q]/sqrt( sum(square(V[,q])) );
        }
        return V;
    }
    real sign(real x){
        if (x < 0.0)
            return -1.0;
        else
            return 1.0;
    }
    matrix Householder (int k, matrix V) {
        // Householder transformation corresponding to kth column of V
        int D = rows(V);
        vector[D] v = V[, k];
        matrix[D,D] H;
        real sgn = sign(v[k]);
        
        //v[k] +=  sgn; //v[k]/fabs(v[k]);
        v[k] += v[k]/fabs(v[k]);
        H = diag_matrix(rep_vector(1, D)) - (2.0 / dot_self(v)) * (v * v');
        H[k:, k:] = -sgn*H[k:, k:];
        return H;
    }
    matrix[] H_prod_right (matrix V) {
        // Compute products of Householder transformations from the right, i.e. backwards
        int D = rows(V);
        int Q = cols(V);
        matrix[D, D] H_prod[Q + 1];
        H_prod[1] = diag_matrix(rep_vector(1, D));
        for (q in 1:Q)
            H_prod[q + 1] = Householder(Q - q + 1, V) * H_prod[q];
        return H_prod;    
    }
    matrix orthogonal_matrix (int D, int Q, vector v) {
        matrix[D, Q] V = V_low_tri_plus_diag(D, Q, v);
        // Apply Householder transformations from right
        matrix[D, D] H_prod[Q + 1] = H_prod_right(V);
        return H_prod[Q + 1][, 1:Q];    
    }
}
data{
    int<lower=0> N;
    int<lower=1> D;
    int<lower=1> Q;
    vector[D] Y[N];
}
transformed data{
    vector[D] mu = rep_vector(0, D);
}
parameters{
    //vector[D*(D+1)/2 + D] v;
    vector[D*Q - Q*(Q-1)/2] v;
    positive_ordered[Q] sigma;
    
    //vector[D] mu;
    real<lower=0> sigma_noise;
}
transformed parameters{
    matrix[D, Q] W;
    cholesky_factor_cov[D] L;
    
    {
        matrix[D, Q] U = orthogonal_matrix(D, Q, v);
        matrix[D, D] K;
        
        W = U*diag_matrix(sigma);
        
        K = W*W';
        for (d in 1:D)
            K[d, d] = K[d,d] + square(sigma_noise) + 1e-14;
        L = cholesky_decompose(K);
    }
}
model{
    //mu ~ normal(0, 10);
    sigma_noise ~ normal(0,0.5);
    
    //v ~ normal(0,1);
    {
        matrix[D, Q] V = V_low_tri_plus_diag_unnormed(D, Q, v);
        for (q in 1:Q) {
            real r = sqrt(dot_self(V[,q]));
            r ~ gamma(100,100);
            target += -log(r)*(D-q);
        }
    }
    
    //prior on sigma
    target += -0.5*sum(square(sigma)) + (D-Q-1)*sum(log(sigma));
    for (i in 1:Q)
        for (j in (i+1):Q)
            target += log(square(sigma[Q-i+1]) - square(sigma[Q-j+1]));
    target += sum(log(2*sigma));
    
    Y ~ multi_normal_cholesky(mu, L);   
}
generated quantities {
    matrix[D, Q] U_n = orthogonal_matrix(D, Q, v);
    matrix[D, Q] W_n;
    
    //for (q in 1:Q)
        //if (U_n[1,q] < 0){
            //U_n[,q] = -U_n[,q];
        //}
    W_n = U_n*diag_matrix(sigma);
}

