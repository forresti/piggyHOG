function compile(opt, verb, mex_file)
% Build MEX source code.
%   All compiled binaries are placed in the bin/ directory.
%
%   Windows users: Windows is not yet supported. You can likely 
%   get the code to compile with some modifications, but please 
%   do not email to ask for support.
%
% Arguments
%   opt   Compile with optimizations (default: on)
%   verb  Verbose output (default: off)

if ispc
  error('This code is not supported on Windows.');
end

if nargin < 1
  opt = true;
end

if nargin < 2
  verb = false;
end

% Start building the mex command
mexcmd = 'mex -outdir bin';

% Add verbosity if requested
if verb
  mexcmd = [mexcmd ' -v'];
end

% Add optimizations if requested
if opt
  mexcmd = [mexcmd ' -O'];
  mexcmd = [mexcmd ' CXXOPTIMFLAGS="-O3 -DNDEBUG -mavx -msse2avx -mtune=corei7-avx"'];
  mexcmd = [mexcmd ' LDOPTIMFLAGS="-O3"'];
else
  mexcmd = [mexcmd ' -g'];
end

% Turn all warnings on
mexcmd = [mexcmd ' CXXFLAGS="\$CXXFLAGS -Wall"'];
mexcmd = [mexcmd ' LDFLAGS="\$LDFLAGS -Wall"'];

if nargin < 3
  % Build feature vector cache code
  fv_compile(opt, verb);
  % Build the star-cascade code
  %cascade_compile(opt, verb);

  eval([mexcmd ' features/resize.cc']);
  eval([mexcmd ' features/features.cc']);
else
  eval([mexcmd ' ' mex_file]);
end
