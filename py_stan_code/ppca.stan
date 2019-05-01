data{
    int<lower=0> N;
    int<lower=1> D;
    int<lower=1> Q;
    vector[D] Y[N];
}
transformed data{
}
parameters {
    matrix[D, Q] W;
    real<lower=0> sigma_noise;
	vector[D] mu;
}
transformed parameters {
    cholesky_factor_cov[D] L;
    {
        //matrix[D, D] K = W*W'; // tcrossprod(matrix x)
        matrix[D, D] K = tcrossprod(W);
        for (d in 1:D)
            K[d,d] += square(sigma_noise) + 1e-14;
        L = cholesky_decompose(K);
    }
}
model{
	mu ~ normal(0, 10);
    sigma_noise ~ normal(0,1);
    to_vector(W) ~ normal(0,1);
    
    Y ~ multi_normal_cholesky(mu, L);
}
