\ProvidesClass{winnower}

\LoadClass{article}

%\RequirePackage{longtable}%

\RequirePackage{amsmath}
\RequirePackage[noblocks]{authblk}
\RequirePackage{natbib}
\RequirePackage{hyperref} 
\RequirePackage{amssymb}
\RequirePackage{graphicx}

%  ****************************************
%  *            MACROS               *
%  ****************************************


\def\acknowledgments{\vskip 5.8mm plus 1mm minus 1mm}
\let\acknowledgements=\acknowledgments                  % second spelling

%  ****************************************
%  *            PAGE LAYOUT               *
%  ****************************************

% Page size, spacing parameters, etc. 

\RequirePackage{fullpage}
\RequirePackage{fancyhdr}

\pagestyle{fancy}
\headheight=.3in

\rhead[]{}
\lhead[The Winnower]{Technicolor Rennes}
%\cfoot[]{}
%\rfoot[\thepage]{\thepage}

\renewcommand{\headrulewidth}{0.6pt}
\renewcommand{\footrulewidth}{0.6pt}

\def\eqsecnum{
    \@newctr{equation}[section]
    \def\theequation{\hbox{\normalsize\arabic{section}-\arabic{equation}}}}
