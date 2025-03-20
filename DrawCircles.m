function drawCirclesAndCalculateStats(imagePath)
    %Function to draw circles over MRI images
    %PRESS ESC AFTER DRAWING EACH CIRCLE
    %Function outputs number of circles, inter-centroid distances
    %and circles diameters
    %Jesus Fajardo 
    %jesuseff@wayne.edu
    %03/20/25
    % Load the image
    img = imread(imagePath);
    imshow(img);
    hold on;

    % Initialize variables
    circles = [];
    diameters = [];
    centroids = [];

    % Loop to draw circles manually
    while true
        h = imellipse(gca, 'PositionConstraintFcn', @(pos) pos);
        wait(h);
        pos = getPosition(h);

        % Calculate diameter and centroid
        diameter = mean([pos(3), pos(4)]);
        centroid = [pos(1) + pos(3)/2, pos(2) + pos(4)/2];

        % Store values
        diameters = [diameters; diameter];
        centroids = [centroids; centroid];

        % Draw the circle
        rectangle('Position', pos, 'Curvature', [1, 1], 'EdgeColor', 'r', 'LineWidth', 2);

        % Ask if user wants to draw another circle
        choice = questdlg('Do you want to draw another circle?', 'Continue', 'Yes', 'No', 'Yes');
        if strcmp(choice, 'No')
            break;
        end
    end

    % Calculate statistics
    numCircles = size(centroids, 1);
    if numCircles > 1
        distances = pdist(centroids);
        avgDistance = mean(distances);
        stdDistance = std(distances);
    else
        distances = [];
        avgDistance = 0;
        stdDistance = 0;
    end
    avgDiameter = mean(diameters);
    stdDiameter = std(diameters);

    % Output results
    fprintf('Number of circles: %d\n', numCircles);
    fprintf('Average inter-centroid distance: %.2f\n', avgDistance);
    fprintf('Standard deviation of inter-centroid distance: %.2f\n', stdDistance);
    fprintf('Average circle diameter: %.2f\n', avgDiameter);
    fprintf('Standard deviation of circle diameter: %.2f\n', stdDiameter);

    % Save variables to .mat file
    [~, name, ~] = fileparts(imagePath);
    save([name, '.mat'], 'distances', 'centroids', 'diameters');
end
