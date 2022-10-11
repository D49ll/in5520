%% TASK 1
%
% The exercise objective is to highlight only the zebras in the attached images.
% Because of the zebras' prominent texture, we can do so by using GLCMs.
% Let's slide a window over an image of a zebra, sending its content
% to the GLCM function. The returned GLCMs are used to derive features such
% as contrast, variance and entropy - which in turn will be used to
% threshold and hopefully highlight only the zebras. 

% GLCM PARAMETERS
dx = 1;           % x-direction
dy = 0;           % y-direction
normalise = true; % Normalise the GLCM?
symmetric = false; % Create symmetric GLCM?
G = 16;            % nof. gray scale levels
windowSize = 31;  % window size 9

% IMAGE PREPARATIONS
img_orig = imread('zebra_1.tif');       % Load the image. NB! Only zebra_1.tif has 256 gray levels
img = uint8(floor(double(img_orig)/G)); % Reduce nof. gray scale levels to G = 16
figure,imshow(img,[])

%img = img_orig;

%however the maximum is 14 after reducing the number of graylevels
% use this in order to get more contrast in the image and use all gray
img = histeq(img,G); % we equalize in order to use all the graylevels
img = uint8(round(double(img) * (G-1) / double(max(img(:))))); % We divide by the largest value and then multiply by G-1 in order to get use all of the range avaliable
figure,imshow(img,[])

[m,n] = size(img);              % Size of original image
halfSize = floor(windowSize/2); % Size of "half" the window

% Expand the original image with a zero border
imgWithBorders = zeros(m+windowSize-1, n+windowSize-1);
imgWithBorders(halfSize:end-halfSize-1,halfSize:end-halfSize-1) = img;
[M,N] = size(imgWithBorders); % Size of image with borders

% Declare feature matrices
C = zeros(size(img)); % Contrast matrix
V = zeros(size(img)); % Variance matrix
E = zeros(size(img)); % Entropy matrix

ii = repmat((0:(G-1))', 1, G);
jj = repmat( 0:(G-1)  , G, 1);

for i = 1+halfSize:N-halfSize-1 %1:size(img, 2)-windowSize
    for j = 1+halfSize:M-halfSize-1 %1:size(img, 1)-windowSize
        
        %Code that is commented out in this block can be run without the
        %loop in that case you have to select the lines and press F9 in
        %order to run just that selection.
        % You can use this to see how the weights of the different 2. order
        % moments behave given a position in the image (j,i)
        % j=385;
        % i=289;
        
        % Retrieve window
        window = imgWithBorders(j-halfSize:j+halfSize,i-halfSize:i+halfSize);
        % figure,imshow(window,[]),title('Window in image')
        
        % Pass window to GLCM function with specified parameters
        K = glcm(window, G, dx, dy, symmetric, normalise);
        % figure,imshow(K,[]),title('GLCM')
        
        % Compute variance feature.
        mu = mean(window(:));
        variance_weight = (ii-mu).^2;
        % figure,surf(variance_weight)
        % The variance has the form: sum_x sum_y (x-mu)^2 * GLCM = sum_x sum_y (y-mu)^2 * GLCM
        V(j-halfSize,i-halfSize) = sum(sum(K .* (variance_weight)));
        
        
        % Compute contrast feature.
        contrast_weight = (ii - jj).^2;
        % figure, surf(contrast_weight)
        % Contrast has the form: sum_x sum_y (x-y) * GLCM
        C(j-halfSize,i-halfSize) = sum(sum(K .* (contrast_weight)));
        
        % Compute entropy feature, which is a value based feature
        E(j-halfSize,i-halfSize) = entropy(K);
    end
    
    % Print calculation progress
    if mod(i,50) == 0 && 100/size(img, 2)*i < 100
        fprintf('Progress: %.1f%%\n', 100/size(img, 2)*i);
    end
end
fprintf('Progress: 100.0%%\nDONE!\n');

%% TASK 2
% Use simple thresholding to mask out the zebras
%
% Here, you can find threshold values using Otsu's method
% thresh_contrast = graythresh(C);
% thresh_variance = graythresh(V);
% thresh_entropy = graythresh(E);

% Or adjust the threshold values manually, as below (might give more accurate results)
thresh_contrast = 0.2;
thresh_variance = 0.3;
thresh_entropy = 0.75;

% Display the results
figure;%clf
subplot(211)
imshow(V, []); title('GLCM Variance');
subplot(212)
%This is threshold: V > (max(V(:)) * thresh_variance)
%then we want to show the image part so we convert the thresholded image to
%uint8  so we can elementwise multiply with the thresholded image.
imshow(img.*uint8(V > (max(V(:)) * thresh_variance)),[]);
title('Image thresholded with GLCM Variance');

figure;%clf
subplot(211)
imshow(C, []); title('GLCM Contrast');
subplot(212)
imshow(img.*uint8(C > (max(C(:)) * thresh_contrast)),[]);
title('GLCM Contrast thresholded');

figure;%clf
subplot(211)
imshow(E, []); title('GLCM Entropy' );
subplot(212)
imshow(img.*uint8(abs(E) > (max(abs(E(:))) * thresh_entropy)),[]);
title('GLCM Entropy tresholded');


% combine the different feature images to see what happens
AND_img = img.*uint8((V > (max(V(:)) * thresh_variance)).*(abs(E) > (max(abs(E(:))) * thresh_entropy)).*(C > (max(C(:)) * thresh_contrast)));
OR_img = img.*uint8(V > (max(V(:)) * thresh_variance))+img.*uint8(abs(E) > (max(abs(E(:)))))+img.*uint8(C > (max(C(:)) * thresh_contrast));

figure;
subplot(211)
imshow(AND_img, []); title('AND between all thresholds' );
subplot(212)
imshow(OR_img,[]);
title('OR between all thresholds');

%% TASK 3

% Compare your result with the first order texture measures: variance and entropy by using
% the Matlab functions: stdfilt and entropyfilt. 

% Remember that the variance is the square of the standard deviation
std_var = stdfilt(img, ones(windowSize)).^2;
figure; subplot(211); imshow(std_var, []); title('Variance');
subplot(212), imshow(img.*uint8(std_var > (max(std_var(:)) * 0.25)),[]);
title('Tresholded Variance')

% See page 532 in Gonzales and woods for a definition on entropy ("average
% information")
std_ent = entropyfilt(img, ones(windowSize));
figure; subplot(211); imshow(std_ent, []); title('Entropy' );
subplot(212), imshow(img.*uint8(std_ent > (max(std_ent(:)) * 0.6)),[])
title('Tresholded entropy')

%
%

% Variance on image after histeq
img_std = histeq(img,G);
img_std = uint8(round(double(img_std) * (G-1) / double(max(img_std(:)))));
std_var_of_img_std = stdfilt(img_std, ones(windowSize)).^2;
figure, subplot(211), imshow(V, []), title('GLCM Variance');
subplot(212), imshow(std_var_of_img_std, []), title('Variance');

