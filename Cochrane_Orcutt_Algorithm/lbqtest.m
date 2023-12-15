function varargout = lbqtest(Data,varargin)
%LBQTEST Ljung-Box Q-test for residual autocorrelation
%
% Syntax:
%
%   [h,pValue,stat,cValue] = lbqtest(res)
%   StatTbl = lbqtest(Tbl)
%   [...] = lbqtest(...,param,val,...)
%
% Description:
%
%   The "portmanteau" test of Ljung and Box assesses the null hypothesis
%   that a series of residuals exhibits no autocorrelation for a fixed
%   number of lags L, against the alternative that some autocorrelation
%   coefficient rho(k), k = 1, ..., L, is nonzero. The test statistic is
%
%                  L
%   	Q = T(T+2)Sum(rho(k)^2/(T-k)),
%                 k=1
%
%   where T is the sample size, L is the number of autocorrelation lags,
%   and rho(k) is the sample autocorrelation at lag k. Under the null, the
%   asymptotic distribution of Q is chi-square with L degrees of freedom.
%
% Input Arguments:
%
%   res - Residual series, specified as a numeric vector. Typically, res
%       contains (standardized) residuals obtained by fitting a model to
%       an observed time series.
%
%   Tbl - Time series data, specified as a table or timetable. Specify a
%       single series for res using the 'DataVariable' parameter.
%
% Optional Input Parameter Name/Value Arguments:
%
%   NAME        VALUE
%
%   'Lags'      Scalar or vector of positive integers indicating the number
%               of lags L used to compute the test statistic. Each element
%               must be less than the effective sample size of res (the 
%               number of nonmissing values in res). The default value is 
%               min[20,T-1], where T is the effective sample size of res.
%
%   'Alpha'     Scalar or vector of nominal significance levels for the
%               tests. Elements must be greater than zero and less than
%               one. The default value is 0.05.
%
%   'DoF'       Scalar or vector of degrees-of-freedom for the asymptotic
%               chi-square distributions of the test statistics. Elements 
%               must be positive integers less than or equal to the 
%               corresponding element of lags. The default value is the 
%               value of lags.
%
%   'DataVariable' Variable in Tbl to use for res, specified as a name in
%               Tbl.Properties.VariableNames. Variable names are character
%               vectors, string scalars, integers or logical vectors. The
%               default is the last variable in Tbl.
%
%   Scalar parameter values are expanded to the length of any vector value
%   (the number of tests). Vector values must have equal length. If any
%   value is a row vector, all outputs are row vectors.
%
% Output Arguments:
%
%   h - Vector of Boolean decisions for the tests, with length equal to the
%   	number of tests. Values of h equal to 1 indicate rejection of the
%       null of no autocorrelation in favor of the alternative. Values of h
%       equal to 0 indicate a failure to reject the null.
%
%   pValue - Vector of p-values of the test statistics, with length equal
%       to the number of tests.
%
%   stat - Vector of test statistics, with length equal to the number of
%       tests.
%
%   cValue - Vector of critical values for the tests, determined by Alpha,
%       with length equal to the number of tests.
%
%   StatTbl - When input is Tbl, outputs h, pValue, stat, and cValue are
%       returned in table StatTbl, with a row for each test. The table also
%       contains variables for the parameter settings of 'Lags', 'Alpha'
%       and 'DoF'.
%
% Notes:
%
%   o The input lags affects the power of the test. If L is too small, the
%     test will not detect high-order autocorrelations; if it is too large,
%     the test will lose power when significant correlation at one lag is
%     washed out by insignificant correlations at other lags. The default
%     value of min[20,T-1] is suggested by Box, Jenkins, and Reinsel [1].
%     Tsay [4] cites simulation evidence that a value approximating log(T)
%     provides better power performance.
%
%   o When res is obtained by fitting a model to data, the degrees of
%     freedom are reduced by the number of estimated coefficients,
%     excluding constants. For example, if res is obtained by fitting an
%     ARMA(p,q) model, DoF should be L-p-q.
%
%   o LBQTEST does not test directly for serial dependencies other than
%     autocorrelation, but it can be used to identify conditional
%     heteroscedasticity (ARCH effects) by testing squared residuals. See,
%     e.g., McLeod and Li [3]. Engle's test, implemented by ARCHTEST, tests
%     for ARCH effects in a residual series directly.
%
%   o LBQTEST treats missing (NaN-valued) residuals as observations that
%     are "missing completely at random."
%
% Example:
%
%   % Test exchange rates for autocorrelation, ARCH effects:
%
%   load Data_MarkPound
%   returns = price2ret(Data);
%   residuals = returns-mean(returns);
%   h1 = lbqtest(residuals)
%   h2 = lbqtest(residuals.^2)
%
% References:
%
%   [1] Box, G.E.P., G.M. Jenkins, and G.C. Reinsel. Time Series Analysis:
%       Forecasting and Control. 3rd ed. Upper Saddle River, NJ:
%       Prentice-Hall, 1994.
% 
%   [2] Gourieroux, C. ARCH Models and Financial Applications. New York:
%       Springer-Verlag, 1997.
%
%   [3] McLeod, A.I. and W.K. Li. "Diagnostic Checking ARMA Time Series
%       Models Using Squared-Residual Autocorrelations." Journal of Time
%       Series Analysis. Vol. 4, 1983, pp. 269-273.
%
%   [4] Tsay,R.S. Analysis of Financial Time Series. Hoboken, NJ: John
%       Wiley & Sons, Inc., 2005.
%
% See also AUTOCORR, ARCHTEST.

% Copyright 2022 The MathWorks, Inc.

isTabular = istable(Data) || istimetable(Data);

% Parse inputs and set defaults:

parseObj = inputParser;
parseObj.addRequired ('Data',...
                      @(x)validateattributes(x,{'double','table','timetable'},{'nonempty','2d'}));
parseObj.addParameter('Lags',[],...
                      @(x)validateattributes(x,{'double'},{'nonempty','integer','vector','positive'}));
parseObj.addParameter('Alpha',0.05,...
                      @(x)validateattributes(x,{'double'},{'nonempty','vector','>',0,'<',1}));
parseObj.addParameter('DoF',[],...
                      @(x)validateattributes(x,{'double'},{'nonempty','vector','positive','integer'}));
parseObj.addParameter('DataVariable',[],...
                      @(x)validateattributes(x,{'double','logical','char','string'},{'vector'}));

try
    
  parseObj.parse(Data,varargin{:});
  
catch ME
    
  throwAsCaller(ME)
  
end

Data  = parseObj.Results.Data;
lags  = parseObj.Results.Lags;
alpha = parseObj.Results.Alpha;
dof   = parseObj.Results.DoF;

varSpec = parseObj.Results.DataVariable;

% Select res with 'DataVariable':

if isnumeric(Data)
    
	res = Data;
    
    if isvector(res) && ~isempty(varSpec)
        
        warning(message('econ:lbqtest:DataVariableUnused'))
        
    end
    
else % Tabular data

    if ~isempty(varSpec)

        try

            res = Data(:,varSpec);

        catch ME

            throwAsCaller(ME)

        end

    else

        res = Data(:,end); % Default

    end
    
    try

        internal.econ.TableAndTimeTableUtilities.isTabularFormatValid(res,'res')
        internal.econ.TableAndTimeTableUtilities.isTabularDataSinglePath(res,'res')

    catch ME

        throwAsCaller(ME)

    end

    res = table2array(res);
    res = double(res);

end

if ~isvector(res)

    error(message('econ:lbqtest:NonVectorInput'))

end

% Set default lags, dof:

T = sum(~isnan(res));      % Effective sample size
defaultLags = min(20,T-1); % Recommended in [1]

if isempty(lags)
    
    lags = defaultLags;
    
elseif lags > (T-1)
    
    error(message('econ:lbqtest:LagsTooLarge'))
    
end

if isempty(dof)
    
    dof = lags; % Default DoF
    
end

% Check parameter values for commensurate lengths, expand scalars, and
% convert all variables to columns:

[numTests,rowOutput,lags,alpha,dof] = sizeCheck(lags,alpha,dof);
res = res(:);

if any(dof > lags)
   error(message('econ:lbqtest:DofTooLarge'))
end

% Compute the sample ACF out to the largest lag:

maxLag = max(lags);
ACF = autocorr(res,10,20); % Lags 0, 1, ..., maxLag
ACF = ACF(2:end);                     % Strip off ACF at lag 0

% Compute Q-statistics to the largest lag; keep only those requested:

idx = (T-(1:maxLag))';
stat = T*(T+2)*cumsum((ACF.^2)./idx);
stat = stat(lags);

% Compute p-values:

pValue = 1-chi2cdf(stat,dof);

% Compute critical values, if requested:

if isTabular
    
    cValue = chi2inv(1-alpha,dof);
    
else

    if nargout >= 4

       cValue = chi2inv(1-alpha,dof);

    else

       cValue = [];

    end

end

% Perform the test:

h = (alpha >= pValue);

% Display outputs as row vectors if lags is a row vector:

if isTabular
    
    % Outputs are rows of Tbl
    
else % Separate outputs

    if rowOutput

       h = h';
       pValue = pValue';
       stat = stat';

       if ~isempty(cValue)

          cValue = cValue';

       end

    end

end

% Create output table:

if isTabular
    
    rowNames = "Test " + num2str((1:numTests)');
    StatTbl = table(h,pValue,stat,cValue,lags,alpha,dof,...
                    'VariableNames',{'h','pValue','stat','cValue','Lags','Alpha','DoF'},...
                    'RowNames',rowNames);
    
end

% Assign outputs to varargout:

if isTabular
    
    nargoutchk(0,1)
    
    varargout{1} = StatTbl;
    
else
    
    nargoutchk(0,4)
    
    varargout{1} = h;
   	varargout{2} = pValue;
    varargout{3} = stat;
    varargout{4} = cValue;
    
end

%-------------------------------------------------------------------------
% Check parameter values for commensurate lengths, expand scalars, and
% convert all variables to columns
function [numTests,rowOutput,varargout] = sizeCheck(varargin)

% Initialize outputs:

numTests = 1;
rowOutput = false;

% Determine vector lengths, row output flag:

for i = 1:nargin
        
    ivar = varargin{i};
    iname = inputname(i);
    
    paramLength.(iname) = length(ivar);
    numTests = max(numTests,paramLength.(iname));
    
    if ~isscalar(ivar)
        rowOutput = rowOutput || (size(ivar,1) == 1);
    end    
    
end

% Check for commensurate vector lengths:

for i = 1:(nargin-1)
    iname = inputname(i);
    for j = (i+1):nargin
        jname = inputname(j);
        if (paramLength.(iname) > 1) && (paramLength.(jname) > 1) ...
            && (paramLength.(iname) ~= paramLength.(jname))
        
            error(message('econ:lbqtest:ParameterSizeMismatch', iname, jname))
              
        end        
    end
end

% Expand scalars:

for i = 1:nargin
    
    ivar = varargin{i};
    if paramLength.(inputname(i)) == 1
        varargout{i} = ivar(ones(numTests,1)); %#ok
    else
        varargout{i} = ivar(:);  %#ok Column output 
    end
    
end

end % sizeCheck

end % LBQTEST