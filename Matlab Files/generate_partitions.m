getGraph('sparse_graph.txt')

function graph = getGraph(string)
    % read in array from file
    array = readSimMatrix(string);
    %array
    disp('Dimensions of Array');
    disp(size(array));
    disp('Read in matrix, converting to list of connections with weights')
    
%     % count number of edges
%     [m,n] = size(array);
%     numConnections = 0;
%     for i = 1:m
%         for j = 1:n
%             if i >= j
%                 continue
%             end
%             if array(i,j) ~= 0
%                 numConnections = numConnections + 1;
%             end
%         end
%     end
%     %numConnections
%     
%     % allocate edges array
%     listOfConnections = zeros(numConnections, 3);
%     size(listOfConnections)
%     idx = 1;
%     for i = 1:m
%         for j = 1:n
%             if i == j
%                 continue
%             end
%             if array(i,j) ~= 0
%                 row = [i-1, j-1, array(i,j)];
%                 listOfConnections(idx, :) = row;
%                 idx = idx+1;
%             end
%         end
%     end
%     %listOfConnections
%     disp('Finished converting to list of connections with weights, starting stability')

    
    % generate T's for markov
    T = 10.^(-2:0.01:2);
    
    % run stability
    [S, N, VI, C] = stability(array, T, 'L', 10);
    save('Stabilities.mat', 'S')
    save('Number of Communities.mat', 'N')
    save('Variation.mat', 'VI')
    save('Cluster Labels.mat','C')
    
end

% function list = convertAdjacencyMatrix(array)
%     [m, n] = size(array);
%     list = {};
%     
% end

function [array,count] = readSimMatrix(string)
   fileID = fopen(string, 'r');

   % count number of lines to preallocate array
   line = fgetl(fileID);
   count = 0;
   while ischar(line)
       line = fgetl(fileID);
       count = count+1;
   end
   
   % allocate array
   disp('Number of lines in file: ')
   disp(count)
   array = zeros(count, count);
   
   % read in line by line
   frewind(fileID)
   line = fgetl(fileID);
   index = 1;
   while ischar(line)
       
       % read row and convert to doubles
       rowStringArr = strsplit(line, ',');
       row = str2double(rowStringArr);
       
       % set row in array
       array(index, :) = row;
       
       % update variables
       line = fgetl(fileID);
       index = index + 1;
   end
   fclose(fileID);
end

