%% Deconvolution and co 

clear all
close all

% Handling the image

I = double(imread('cameraman.tif'));

figure, imagesc(I), colormap('gray'), axis image

% Adding noise for denoising purposes

% Gaussian noise + gaussian blur
k=5;
gamma = 2*k+1; % size of kernel
sigma = 2; % sqrt variance of kernel
[conv_op, conv_adj_op, kA] = createGaussianBlurringOperator(size(I),gamma,sigma);

var = 5; % noise variance
I_nse = conv_op(I) + var * randn(size(I));

% visualize blurred and noisy image
figure; imagesc(I_nse); colormap('gray'); axis image;


%% Deconvolution using Van-Cittert iterations

% specifiy amount of Van-Cittert iterations
iterations=15;

% perform deconvolution with Van-Cittert iterations
uv = deconv_van_cittert(I_nse,kA,iterations);

% visualize deconvolution results
figure; imagesc(uv); colormap('gray'); axis image;
drawnow


%% Deconvolution with TV-regularization and changing operators

% specifiy regularization parameter
lambda = 0.1;

% perform deconvolution using TV regularization
u = deconv_TV_pd(I_nse,kA,lambda);

% visualize deconvolution results
figure; imagesc(u); colormap('gray'); axis image;
drawnow

