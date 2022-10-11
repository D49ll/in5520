% Computes the gray level co-occurrence matrix for a given image window
%
% Arguments:    window     - the image window of size NxM
%               G          - the number of gray levels to use
%               dx         - GLCM x-direction and step size
%               dy         - GLCM y-direction and step size
%               symmetric  - compute symmetric GLCM if true
%               normalise  - normalise the GLCM if true, so that
%                            sum(GLCM(:)) = 1

function GLCM = glcm(window, G, dx, dy, symmetric, normalise)
    
    % Initialise co-ocurrent matrix of size GxG
    GLCM = zeros(G);
    [N,M] = size(window);

    for i = 1:N
        for j = 1:M

            % Skip iteration if we exceed the window boundaries
            if i+dy > N || i+dy < 1 || j+dx > M || ...
                    j+dy < 1 || i + dx < 1 || j + dx < 1
               continue
            end

            % Get the values of the pixel neighbours
            gray1 = window(i,j);
            gray2 = window(i+dy,j+dx);
            % Accumulute the transition between the pixels
            GLCM(gray1+1, gray2+1) = GLCM(gray1+1, gray2+1) + 1; 
        end
    end
    
    % Make the GLCM symmetric
    if symmetric
        GLCM = GLCM + transpose(GLCM);
    end
    % Normalise the GLCM
    if normalise
        GLCM = GLCM / sum(sum(GLCM));
    end  
