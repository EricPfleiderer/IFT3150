clear all
%Set baseline model parameters
PA.a1 =  1.183658646441553.*30 ;
PA.a2 = 1.758233712464858.*30  ;
PA.d1 =  0;
PA.d2 = 0.539325116600707.*30;
PA.d3 = PA.d2;
PA.tau = (33.7/24 - 30/PA.a2)./30; % From Sato 2016 mean intermitotic time - the expected time in G1- gives a lower bound on a_2. 
PA.IntermitoticSD = 6.7/24/30;  % From Sato 2016 SD on intermitotic time

%Distribution Specific Parameters
PA.N = round(PA.tau.^2./PA.IntermitoticSD^2); % round((33.7./24)^2./(6.7./24));
PA.TransitRate = PA.N./PA.tau; % Transit rate across compartments
PA.d3Hat = PA.N./PA.tau.*(exp(PA.d3.*PA.tau./(PA.N+1))-1);
PA.d3HatR = PA.d3Hat;

% Time interval
tf = 3;
tstep = floor(tf./20);
totaltime = [0 tf];


% Viral Therapy
PA.kappa = 3.534412642851458.*30; %  3.686918580939023.*30;
PA.delta =  4.962123414821151.*30; %4.598678773505142.*30;
PA.alpha =  0.008289097649957; % 0.661864324862171.*30;
PA.omega =  9.686308020782763.*30; %3.017878576601330.*30;
PA.eta12 =  0.510538277701167;% 0.056468694630527; % virion half effect contact rate

% Cytokine Parameters
PA.CprodHomeo = 0.00039863.*30;   %Homeostatic cytokine production
PA.CprodMax =  1.429574637713578.*30;   % Maximal cytokine production rate
PA.C12 = 0.739376299393775.*30 ; % Half effect in cytokine production
PA.kel = 0.16139.*30;   % elimination rate of cytokine

%Immune Parameters
PA.kp = 0.050.*30; %.*30; %0.35.*30;  % contact rate with phagocytes
PA.kq = 10;  % factor in denominator of Q phagocytosis term 
PA.ks = PA.kq; % factor in denominator of S phagocytosis term
PA.P12 = 5;   % Half effect in cytokine driven phagocyte production
PA.gammaP = 0.35.*30; %.*30; % From Barrish 2017 PNAS elimination rate of phagocyte
PA.Kcp = 4.6754.*30; %.*30; % From Barrish 2017 PNAS conversion of cytokine into phagocyte

% Immune Steady State
PA.CStar = PA.CprodHomeo./PA.kel;
PA.PStar = (1./PA.gammaP).*(PA.Kcp.*PA.CStar./(PA.P12+PA.CStar));

%Resistant Parameters
PA.nu = 1e-10; %0.00000000005;% Mutation percentage
PA.a1R = PA.a1;
PA.a2R = PA.a2;
PA.d1R = PA.d1;
PA.d2R = PA.d2;
PA.d3R = PA.d3;
PA.kappaR = PA.kappa;
    
%% Calculate the initial conditions
% Calculate the cell cycle duration
TotalTime = 1/PA.a1 +1/(PA.a2+PA.d2)+PA.tau;
PA.TotalCells = 200;
% Initial Conditions
QIC = (1/PA.a1./TotalTime).*PA.TotalCells.*(1-PA.nu); %100;
SIC = (1/(PA.a2+PA.d2)./TotalTime).*PA.TotalCells.*(1-PA.nu);  %100;
TCIC = (PA.tau./TotalTime).*PA.TotalCells.*(1-PA.nu).*ones(1,PA.N)./PA.N; %Transit compartment ICs
NCIC = (PA.tau./TotalTime).*PA.TotalCells.*(1-PA.nu);
IIC = 0;
VIC = 0;
CIC =  PA.CprodHomeo./PA.kel;
PIC =   PA.Kcp.*CIC./((PA.P12+CIC).*PA.gammaP);
% Resistant Strain ICs
RIC =   (1/PA.a1./TotalTime).*PA.TotalCells.*(PA.nu); %
RSIC =   (1/(PA.a2+PA.d2)./TotalTime).*PA.TotalCells.*(PA.nu); %
ResistantTCIC =   (PA.tau./TotalTime).*PA.TotalCells.*(PA.nu).*ones(1,PA.N)./PA.N; 
ResistantTotalCellsIC =   (PA.tau./TotalTime).*PA.TotalCells.*(PA.nu);
InitialConditions = [QIC,SIC,IIC,VIC,TCIC,CIC,PIC,NCIC,RIC,RSIC,ResistantTCIC,ResistantTotalCellsIC];

%% Set up the Optimizer
PA.AdminNumber = 75; %Number of immunotherapy doses- possible to dose everyday for 2.5 months
PA.ViralAdminNumber = 10; %Number of viral therapy doses - possible to dose every week for 2.5 months
nvars = PA.AdminNumber + PA.ViralAdminNumber;    % Number of doses (variables to optimize over)
% Each dose must be an integer multiple of the baseline dose [0,1,2,...,5].
LB = 0.*ones(1,nvars);   % Lower bound for the optimizer
UB = 4.*ones(1,nvars);    % Upper bound for the optimizer
IntCon = 1:nvars; %The condition that ensures that the optimial solution enforces integer multiple doses of the baseline dose

VirtualPatientParametersInput = load('16052019VirtualPopulation300PatientParameters'); %Load the virtual patients from 16/05/2019
B = struct2array(VirtualPatientParametersInput);
M = length(B(1,:)); %The number of patients
OptimalDose = zeros(M,nvars); %Matrix to store the optimal doses for each patient
for j = 136:137 %For each patient, find the optimal dosing regime
 Param = B([1,2,3,4,6,21,22,23,26],j+1); %load the varied parameters from the virtual patient and update PA.
 C = {Param(1), ... 
      Param(2), ...
      Param(3), ...
      Param(4), ...
      Param(5), ...
      Param(6), ...
      Param(7), ...
      Param(8), ...
      Param(9) };
ParameterNames = {'a1' 'a2' 'd1' 'd2' 'tau' 'kp' 'kq' 'ks' 'Kcp'}; %The name of parameters to be varied
  %Update model parameters
    [PA.(ParameterNames{1}),PA.(ParameterNames{2}),PA.(ParameterNames{3}),PA.(ParameterNames{4})...
     PA.(ParameterNames{5}),PA.(ParameterNames{6}),PA.(ParameterNames{7}),PA.(ParameterNames{8}),PA.(ParameterNames{9})] = C{:}; %update the parameters for this run 
 
    % Resistant parameters
    PA.a1R = PA.a1;
    PA.a2R = PA.a2;
    PA.d1R = PA.d1;
    PA.d2R = PA.d2;
    PA.d3R = PA.d3;
    
    PA.TransitRate = PA.N./PA.tau; %must recalculate transit rate, as the delay varies
    PA.CStar = PA.CprodHomeo./PA.kel;
    PA.PStar = (1./PA.gammaP).*(PA.Kcp.*PA.CStar./(PA.P12+PA.CStar));
    PA.d3Hat = PA.N./PA.tau.*(exp(PA.d3.*PA.tau./(PA.N+1))-1);
    PA.d3HatR = PA.d3Hat;
tic
FObjective = @(x)Updated16052019TreatmentOptimizerObjective(x,PA);  
%Set up the genetic algorithm
opts = optimoptions('ga','EliteCount',15,'MaxStallGenerations',4,'FunctionTolerance',1e-6, 'MaxGenerations',50,'Display','iter','PopulationSize',60);
[x,fval] = ga(FObjective, nvars,[],[],[],[],LB,UB,[],IntCon,opts);
%Store the optimal dosing regime for patient j.
OptimalDose(j,:) = x;
save('OptimizedDoseSavedInLoopTylerLaptop136to150','OptimalDose')
toc
end

