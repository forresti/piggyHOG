

function writeToCsv_withSize(fname, array)
    %csvwrite(fname, array);
    csvwrite(fname, ''); %clear the output file
    internal_writeToCsv(fname, array);

    mySize = size(array);
    if(length(size(array)) == 2)
        sizeStr = sprintf('%d,%d', mySize(1), mySize(2));
    elseif (length(size(array)) == 3)
        sizeStr = sprintf('%d,%d,%d', mySize(1), mySize(2), mySize(3));
    end

    prepend2file(sizeStr, fname, true);
end

%write one line at a time, doing an append.
function internal_writeToCsv(fname, array)
    [depth, height, width] = size(array);
    for x=1:width
        for y=1:height
            %myRow = '';
            for d = 1:depth
                buf = array(:, y, x);
                %C = textscan(sprintf('%s %s\n', buf{:}), '%s', 'delimiter', '\n')

                rowStr = mat2str(buf); %TODO: mat2str(buf, 5) -- '5' = digits of precision
                rowStr = matStr_to_csv(rowStr)

                %dlmwrite(fname, array(:, y, x)', '-append');
            end
        end
    end
end

% [1 2 3 4] -> 1,2,3,4
function matStr = matStr_to_csv(matStr)
    %space to comma
    underscore_loc = findstr(matStr, ' ');
    matStr(underscore_loc) = ',';

    %remove brackets
    bracket_loc = findstr(matStr, '['); 
    matStr(bracket_loc) = ''; 
    bracket_loc = findstr(matStr, ']');
    matStr(bracket_loc) = ''; 
end


%thanks to: http://www.mathworks.com/support/solutions/en/data/1-1BM4K/index.html?product=SL&solution=1-1BM4K
function prepend2file( string, filename, newline )
    % newline: is an optional boolean, that if true will append a \n to the end
    % of the string that is sent in such that the original text starts on the
    % next line rather than the end of the line that the string is on
    % string: a single line string
    % filename: the file you want to prepend to
    tempFile = tempname;
    fw = fopen( tempFile, 'wt' );
    if nargin < 3
    newline = false;
    end
    if newline
    fwrite( fw, sprintf('%s\n', string ) );
    else
    fwrite( fw, string );
    end

    fclose( fw );
    appendFiles( filename, tempFile );
    copyfile( tempFile, filename );
    delete(tempFile);

    % append readFile to writtenFile
    function status = appendFiles( readFile, writtenFile )
    fr = fopen( readFile, 'rt' );
    fw = fopen( writtenFile, 'at' );

    while feof( fr ) == 0
    tline = fgetl( fr );
    fwrite( fw, sprintf('%s\n',tline ) );
    end
    fclose(fr);
    fclose(fw);
    end
end

