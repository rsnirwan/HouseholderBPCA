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
    real<lower=0> kernel_std;
    vector[D] mu;
}
transformed parameters {
    cholesky_factor_cov[D] L;
    {
        matrix[D, D] K;
        
        vector[Q] W_tmp[D];
        for (d in 1:D)
            W_tmp[d] = W[d,]';
        
        K = cov_exp_quad(W_tmp, kernel_std, 1.);
        for (d in 1:D)
            K[d,d] += square(sigma_noise) + 1e-14;
        L = cholesky_decompose(K);
    }
}
model{
    mu ~ normal(0, 10);
    sigma_noise ~ normal(0,1);
    kernel_std ~ normal(0,1);
    to_vector(W) ~ normal(0,1);
    
    Y ~ multi_normal_cholesky(mu, L);
}
