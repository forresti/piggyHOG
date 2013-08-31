

function writeToCsv_withSize(fname, array)
    csvwrite(fname, array);
    internal_writeToCsv(array);

    mySize = size(array);
    if(length(size(array)) == 2)
        sizeStr = sprintf('%d,%d', mySize(1), mySize(2));
    elseif (length(size(array)) == 3)
        sizeStr = sprintf('%d,%d,%d', mySize(1), mySize(2), mySize(3));
    end

    prepend2file(sizeStr, fname, true);
    %prepend2file([num2str], fname, true)
end

%produce a string, where each line is the "d" dimension in array(d,y,x). 
%So, each line in the output is one feature descriptor.
function csvString = internal_writeToCsv(array)
   [depth, height, width] = size(array) 

    csvString = '...'; %placeholder
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

