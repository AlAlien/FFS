function score = FFS(image1, image2)
% ========================================================================
% FFS Index with automatic downsampling, Version 1.0£¬2020.6.23
% Copyright(c) 2019 Chenyang Shi£¬Yandan Lin
% All Rights Reserved.
%
% ----------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is here
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
%----------------------------------------------------------------------
% Shi Chenyang, Lin Yandan. Image Quality Assessment Based on Three 
% Features Fusion in Three Fusion Steps. Symmetry. 2022; 14(4):773.
%----------------------------------------------------------------------
%
%Input : (1) image1: the first image being compared, which is a RGB image
%        (2) image2: the second image being compared, which is a RGB image
%
%Output: sim: the similarity score between two images, a real number
%        
%-----------------------------------------------------------------------

%alpha = 0.515;
alpha = 0.52;
Kv1 = 0.25;%fixed
Kv2 = 0.5 * Kv1;%fixed
Kv3 = 0.5 * Kv1;%fixed
Kc = 270;
Kg1 = 160; 
Kg2 = 90; 

%compute the visual saliency map with color appearance

[rows, cols, junk] = size(image1);

L1 = 0.06 * double(image1(:,:,1)) + 0.63 * double(image1(:,:,2)) + 0.27 * double(image1(:,:,3));
L2 = 0.06 * double(image2(:,:,1)) + 0.63 * double(image2(:,:,2)) + 0.27 * double(image2(:,:,3));
M1 = 0.30 * double(image1(:,:,1)) + 0.04 * double(image1(:,:,2)) - 0.35 * double(image1(:,:,3));
M2 = 0.30 * double(image2(:,:,1)) + 0.04 * double(image2(:,:,2)) - 0.35 * double(image2(:,:,3));
N1 = 0.34 * double(image1(:,:,1)) - 0.60 * double(image1(:,:,2)) + 0.17 * double(image1(:,:,3));
N2 = 0.34 * double(image2(:,:,1)) - 0.60 * double(image2(:,:,2)) + 0.17 * double(image2(:,:,3));

%%%%%%%%%%%%%%%%%%%%%%%%%
% Downsample the image
%%%%%%%%%%%%%%%%%%%%%%%%%
minDimension = min(rows,cols);
F = max(1,round(minDimension / 256));
aveKernel = fspecial('average',F);

aveL1 = conv2(L1, aveKernel,'same');
aveL2 = conv2(L2, aveKernel,'same');
L1 = aveL1(1:F:rows,1:F:cols);
L2 = aveL2(1:F:rows,1:F:cols);

FF = alpha * (L1 +L2); % Fusion

aveM1 = conv2(M1, aveKernel,'same');
aveM2 = conv2(M2, aveKernel,'same');
M1 = aveM1(1:F:rows,1:F:cols);
M2 = aveM2(1:F:rows,1:F:cols);

aveN1 = conv2(N1, aveKernel,'same');
aveN2 = conv2(N2, aveKernel,'same');
N1 = aveN1(1:F:rows,1:F:cols);
N2 = aveN2(1:F:rows,1:F:cols);

% SR Similarity
vs1=spectralResidueSaliency(double(L1));
vs2=spectralResidueSaliency(double(L2));
vs3=spectralResidueSaliency(double((FF)));

VSSimMatrix1 = (2 * vs1 .* vs2 + Kv1) ./ (vs1.^2 + vs2.^2 + Kv1);% VSS of R and D
VSSimMatrix2 = (2 * vs1 .* vs3 + Kv2) ./ (vs1.^2 + vs3.^2 + Kv2);% VSS of R and F
VSSimMatrix3 = (2 * vs3 .* vs2 + Kv3) ./ (vs3.^2 + vs2.^2 + Kv3);% VSS of D and F
VS_HVS=VSSimMatrix1+VSSimMatrix3-VSSimMatrix2;

%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate the gradient map
%%%%%%%%%%%%%%%%%%%%%%%%%

dx = [1 0 -1; 1 0 -1; 1 0 -1] / 3;
dy = dx';

IxL1 = conv2(L1, dx, 'same');     
IyL1 = conv2(L1, dy, 'same');    
gR = sqrt(IxL1.^2 + IyL1.^2);

IxL2 = conv2(L2, dx, 'same');     
IyL2 = conv2(L2, dy, 'same');    
gD = sqrt(IxL2.^2 + IyL2.^2);

IxF = conv2(FF, dx, 'same');
IyF = conv2(FF, dy, 'same');
gF = sqrt(IxF .^ 2 + IyF .^ 2);


% Gradient Similarity (GS)
GS12 = (2 * gR .* gD + Kg1) ./ (gR .^ 2 + gD .^ 2 + Kg1); % GS of R and D
GS13 = (2 * gR .* gF + Kg2) ./ (gR .^ 2 + gF .^ 2 + Kg2); % GS of R and F
GS23 = (2 * gD .* gF + Kg2) ./ (gD .^ 2 + gF .^ 2 + Kg2); % GS of D and F
GS_HVS = GS12 + GS23 - GS13; % HVS-based GS

%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate the FFS
%%%%%%%%%%%%%%%%%%%%%%%%%
CS = (2 * (N1 .* N2 + M1 .* M2) + Kc) ./ (N1 .^ 2 + N2 .^ 2 + M1 .^ 2 + M2 .^ 2 + Kc);

score=0.4*(VS_HVS)+0.4*(GS_HVS)+0.2*(CS);

score = mad( (score(:) .^ 0.5) .^ 0.5 )^0.15;

return;
%===================================
function saliencyMap = spectralResidueSaliency(image)
% ========================================================================
%
%Input : image: an uint8 RGB image with dynamic range [0, 255] for each
%channel
%        
%Output: VSMap: the visual saliency map extracted by the SR algorithm.
%Data range for VSMap is [0, 255]. So, it can be regarded as a common
%gray-scale image.
%        

scale = 0.25; %fixedsc
aveKernelSize =3; %fixed
gauSigma =6; %fixed
gauSize =15; %fixed

inImg = imresize(image, scale);

%%%% Spectral Residual
myFFT = fft2(inImg);
myLogAmplitude = log(abs(myFFT));
myPhase = angle(myFFT);

mySpectralResidual = myLogAmplitude - imfilter(myLogAmplitude, fspecial('average', aveKernelSize), 'replicate');
saliencyMap = abs(ifft2(exp(mySpectralResidual + 1i*myPhase))).^2;

%%%% After Effect
saliencyMap = mat2gray(imfilter(saliencyMap, fspecial('gaussian', [gauSize, gauSize], gauSigma)));
saliencyMap = imresize(saliencyMap,[size(image,1) size(image,2)]);
return;



