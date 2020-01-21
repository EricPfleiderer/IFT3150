%close all
clear all
format long
tic
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
tf = 60;
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


%%  Standard of care dosing
% Dosing parameters
PA.AdminNumber =   180; %Number of immunotherapy doses- possible to dose everyday 
PA.ViralAdminNumber =  26; %Number of viral therapy doses - possible to dose every two weeks for 2.5 months

ImmunoTherapyDoses = ones(1,PA.AdminNumber);
for ii = 1:floor(PA.AdminNumber./2)
    ImmunoTherapyDoses(2.*ii) = 0;
end
ViralTherapyDoses = ones(1,PA.ViralAdminNumber);
for ii = 1:floor(PA.ViralAdminNumber./2)
    ViralTherapyDoses(2.*ii)=0;
end
% Immunotherapy
PA.StartTime = 0;
PA.Offset = 1./30; %Dosed every day (some Admin can be 0)
%Load the dosing strategy for immunotherapy
PA.Admin = 0.*125000.*ImmunoTherapyDoses; % 0.*212500.*ImmunoTherapyDoses; %Amount of immunotherapy administered 
PA.Vol = 7; %volume of absorption
PA.kabs = 6.6311.*30; % absorption rate
PA.AvailFrac =0.85; 

% Viral therapy
PA.ViralStartTime = 0;
PA.ViralOffset = 7./30; %Dosed at a maximum of once weekly
PA.ViralAdmin = 0.*250.*ViralTherapyDoses; %Amount of virus administered (input from the optimizer)
PA.Viralkabs = 20.*30; % absorption rate
PA.ViralAvailFrac = 1;

%% Set up sampling procedure
PA.TotalCells = 200; 
% PA.DeathTime = 2 ;

SizePA = length(struct2array(PA));
%Parameter set up
ParameterNames = {'a1' 'a2' 'd1' 'd2' 'tau' 'kp' 'kq' 'ks' 'Kcp'}; %The name of parameters to be varied
% ub = 1.2; % Set the upper bound for the uniform distribution 
% lb = 1./ub; %set the lower bound for the uniform distribution
VirtualPatientParametersInput = load('16052019VirtualPopulation300PatientParameters');
%VirtualPatientParametersInput = load('30052019VirtualPopulation200PatientParametersForOptimization');
%Load the virtual patients from 16/05/2019
B = struct2array(VirtualPatientParametersInput);
M = 300;
%M = 300; %250; %Number of desired virtual patients less than 300
NoDouble = 0;
TooSlow = 0;
VirtualPatientParameters = zeros(SizePA,M); %array to save the parameters for each virtual patient
DoublingTimes = zeros(3,M+1); % Vector to record the tumour doubling time of each patient
MetastasisTimes = zeros(3,M+1); % vector to record the death time of each patient without treatment (1st row) and with treatment (2nd row)
ProgressionFreeTime = zeros(3,M+1);
% TreatDeathTimes = zeros(2,M+1);
DelayThresholdTime = zeros(1,M+1); 


Stage3DeathTime = 16+ 4.*rand( [1 round(0.3.*M)]); %16+ 4.*rand( [1 round(0.3.*M)]);
Stage4DeathTime = 22 + 4.*rand( [1 round(0.7.*M)]);
%DeathTimeMatrix = load('16052019DeathTimesFor300PatientOPTiMClinicalTrial'); % % For the death times used in finding the kaplan meier curve
%PA.DeathCellBurden = struct2array(DeathTimeMatrix);
 PA.DeathCellBurden = 1e6.*exp( -(2.*PA.a2.*exp(-PA.d3Hat.*PA.tau) - PA.a2-PA.d2).*([Stage3DeathTime,Stage4DeathTime]-18));

for ii = 1:M; 
    %Update the parameters for each virtual patient
Param = B([1,2,3,4,6,21,22,23,26],ii+1); %Load the correct parameters from the matrix of new virtual patients.
 C = {Param(1), ... 
      Param(2), ...
      Param(3), ...
      Param(4), ...
      Param(5), ...
      Param(6), ...
      Param(7), ...
      Param(8), ...
      Param(9) };

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
    
%Distribution Specific Parameters
% PA.N = round(PA.tau.^2./PA.IntermitoticSD^2); % round((33.7./24)^2./(6.7./24));
    
%% Calculate the initial conditions
%Calculate the cell cycle duration
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

%% Death time
PA.DeathTime = PA.DeathCellBurden(ii); 
%%  Solve the Distributed DDE system
%Turn off treatment for control run
PA.Admin = 0.*125000.*ImmunoTherapyDoses; % 0.*212500.*ImmunoTherapyDoses; %Amount of immunotherapy administered 
PA.ViralAdmin = 0.*250.*ViralTherapyDoses; %Amount of virus administered (input from the optimizer)
tic
[solTest] =  WaresDistributedImmunityResistantSolver(totaltime,InitialConditions,PA);
toc
    DoublingTimes(1,ii) = solTest.xe(1,1);
   if length(solTest.xe) < 2 %Test for death without treatment
       MetastasisTimes(1,ii+1) =  2.*tf;
   else
      MetastasisTimes(1,ii+1) = solTest.xe(1,2);
      ProgressionFreeTime(1,ii+1) = solTest.xe(1,end-1);
   end
   DelayThresholdTime(ii+1) = PA.tau./PA.N - (1./(PA.d3Hat+PA.kp.*PA.PStar)).*((2.*PA.a2.*PA.a1/((PA.a1+PA.d1+PA.kp.*PA.PStar).*(PA.a2+PA.d2+PA.kp.*PA.PStar))).^(1/PA.N)-1);
   %Turn off immune treatment for control run and turn on viral treatment
   PA.Admin = 0.*125000.*ImmunoTherapyDoses; %  0.*212500.*ImmunoTherapyDoses; %Amount of immunotherapy administered 
   PA.ViralAdmin = 250.*1.*ViralTherapyDoses; %Amount of virus administered (input from the optimizer)
   [solTreat] = WaresDistributedImmunityResistantSolver(totaltime,InitialConditions,PA); %Calculate death time with viral treatment
   if length(solTreat.xe) < 2 %Test for death with treatment
       MetastasisTimes(2,ii+1) =  2.*tf; %Make the plotting easier- these are censored data points
   else
       MetastasisTimes(2,ii+1) = solTreat.xe(1,end);
        DoublingTimes(2,ii+1) = solTreat.xe(1,1);
        ProgressionFreeTime(2,ii+1) = solTreat.xe(1,end-1);
   end
   %Turn off viral treatment and turn on immune
    PA.Admin = 1.*125000.*ImmunoTherapyDoses; % 212500.*ImmunoTherapyDoses; %Amount of immunotherapy administered 
   PA.ViralAdmin = 0.*250.*ViralTherapyDoses; %Amount of virus administered (input from the optimizer)
   [solImmune] = WaresDistributedImmunityResistantSolver(totaltime,InitialConditions,PA); % Calculate death time with immune treatment
   if length(solImmune.xe) < 2 %Test for death with treatment
       MetastasisTimes(3,ii+1) =  2.*tf; %Make the plotting easier- these are censored data points
   else
       MetastasisTimes(3,ii+1) = solImmune.xe(1,end);
       DoublingTimes(3,ii+1) = solImmune.xe(1,1);
       ProgressionFreeTime(3,ii+1) = solImmune.xe(1,end-1);
   end
ii
end
toc
%% Create survival plots
% Matrix with untreated doubling times and death times for various treatments
OutcomeTimes = [DoublingTimes(1,:);MetastasisTimes];
SortedOutcomeTimes = sortrows(OutcomeTimes');
% For each virtual patient, we simulate when their tumour has reached 2^15 times the initial size
k = 1e6; %Steepness of step function
Time = 0:tf; %Set the time domain of [0,tf] in 1 month increments
NoTreatmentMetastasisTimes = zeros(1,tf);
ViralTreatmentMetastasisTimes = zeros(1,tf);
ImmuneTreatmentMetastasisTimes =   zeros(1,tf);
for ii = 1:tf+1
    NoTreatmentMetastasisTimes(ii) = DeathTimesPlot(Time(ii),MetastasisTimes(1,:),M,k);
    ViralTreatmentMetastasisTimes(ii) = DeathTimesPlot(Time(ii),MetastasisTimes(2,:),M,k);
    ImmuneTreatmentMetastasisTimes(ii) = DeathTimesPlot(Time(ii),MetastasisTimes(3,:),M,k);
end

Fig1 = figure(1); %Kaplan Meier Curve
% g1=plot(Time,NoTreatmentMetastasisTimes,'b','LineWidth',2); set(gca,'fontsize',16) ;
% hold on
g2=plot(Time,ImmuneTreatmentMetastasisTimes,'k','LineWidth',2); set(gca,'fontsize',16) ; 
hold on
g3=plot(Time,ViralTreatmentMetastasisTimes,'r','LineWidth',2); set(gca,'fontsize',16) ; 
set(Fig1, 'Position', [500 250 840 730]);
xlabel('Time (Months)','FontSize',15)
ylabel('Survival','FontSize',15)
legend([g2 g3], 'Immune Treatment', 'Viral Treatment')
hold on

Fig2 = figure(2); %Treatment efficiency 
g1= bar(SortedOutcomeTimes(:,3),1,'FaceColor','r','EdgeColor','w') ; set(gca,'fontsize',16)  %set(gca,'fontsize',16) ; 
%bar([SortedOutcomeTimes(:,2),SortedOutcomeTimes(:,3),SortedOutcomeTimes(:,4)],'stacked'); set(gca,'fontsize',16) ; 
%bar(SortedOutcomeTimes(:,1),[SortedOutcomeTimes(:,2),SortedOutcomeTimes(:,3),SortedOutcomeTimes(:,4)],'stacked'); set(gca,'fontsize',16) ; 
hold on
g2= bar(SortedOutcomeTimes(:,4),1,'FaceColor','k','EdgeColor','w') ; set(gca,'fontsize',16)  ; 
hold on
% g3=bar(SortedOutcomeTimes(:,2),1,'FaceColor','b','EdgeColor','w') ; set(gca,'fontsize',16)  %set(gca,'fontsize',16) ;
% hold on
set(Fig2, 'Position', [500 250 840 730]);
xlim([0 M])
xticks(0:50:M+50)
xticklabels(round([SortedOutcomeTimes(1:50:M+50,1)],1))
xlabel(' Doubling Time (Months)','FontSize',15)
ylim([0 tf])
ylabel('Survival Time (Months)','FontSize',15)
legend([g1 g2],'Viral Treatment', 'Immune Treatment');%, 'No Treatment')
hold on

coefficients = polyfit(SortedOutcomeTimes(2:end,1),( rmmissing( SortedOutcomeTimes(:,3)./(SortedOutcomeTimes(:,4)) ) ), 1);
xFit = linspace(SortedOutcomeTimes(2,1), SortedOutcomeTimes(end,1), 1000);
yFit = polyval(coefficients , xFit);


Fig3 = figure(3);
g1 = scatter(SortedOutcomeTimes(:,1), SortedOutcomeTimes(:,3)./(SortedOutcomeTimes(:,4)),20,'r','*' );
hold on
g2 = plot(xFit,yFit,'b','LineWidth',2);
xlabel(' Doubling Time (Months)','FontSize',15)
ylim([0 4])
ylabel('Ratio of Survival Time','FontSize',15)
%% Calculate survival statistics
DeathTimesUpdated = zeros(3,M);
for ii = 1:3
    for jj = 1:M+1
        if MetastasisTimes(ii,jj) == 2*tf;
        DeathTimesUpdated(ii,jj) = tf;
        else
        DeathTimesUpdated(ii,jj) = MetastasisTimes(ii,jj);
        end
    end
end
ViralTreatmentMeanSurvivial = mean(DeathTimesUpdated(2,:))
ViralTreatmentMedianSurvivial = median(DeathTimesUpdated(2,:))
ViralTreatmentVariance = var(DeathTimesUpdated(2,:))
ImmuneTreatmentMeanSurvival = mean(DeathTimesUpdated(3,:))
ImmuneTreatmentMedianSurvival = median(DeathTimesUpdated(3,:))
ImmuneTreatmentVariance = var(DeathTimesUpdated(3,:))
TStatistic = (ViralTreatmentMeanSurvivial-ImmuneTreatmentMeanSurvival)./( ViralTreatmentVariance./M+ImmuneTreatmentVariance./M)^(1/2)
PValue = 1-tcdf(TStatistic,M-1)
% Log rank statistics
PatientsViral = round( M.*(ViralTreatmentMetastasisTimes(2:end) ) ); %Number of patients without an event
PatientsImmune = round( M.*(ImmuneTreatmentMetastasisTimes(2:end) ) ); %Number of patients without an event
ObservationViral = zeros(1,length(PatientsViral));
ObservationImmune = zeros(1,length(PatientsViral));
for ii = 1 :length(PatientsViral)-1
ObservationViral(ii) = PatientsViral(ii)-PatientsViral(ii+1)  ; %Number of observations
ObservationImmune(ii) = PatientsImmune(ii)-PatientsImmune(ii+1) ; %Number of observations
end
TotalObservation = ObservationImmune+ObservationViral;
TotalPatient = PatientsImmune+PatientsViral;
ExpectedObservationViral = ( (TotalObservation)./( TotalPatient) ).*ObservationViral;
VarianceObservationViral = TotalObservation.*(PatientsViral./TotalPatient).*(1- (PatientsViral./TotalPatient) ).*(TotalPatient-TotalObservation)./(TotalPatient-1);
LogrankZStatistic = sum(ObservationViral-ExpectedObservationViral)./( sum(VarianceObservationViral) )^(1/2);
[h,p] = ztest(LogrankZStatistic,0,1) 
% Time to treatment failure and durable response rate
ProgressionFreeTimeUpdated = zeros(3,M+1);
for ii = 1:3
    for jj = 1:M+1
        if ProgressionFreeTime(ii,jj) == 0;
        ProgressionFreeTimeUpdated(ii,jj) = tf;
        else
        ProgressionFreeTimeUpdated(ii,jj) = ProgressionFreeTime(ii,jj);
        end
    end
end

function DeathPlot = DeathTimesPlot(x,y,M,k)
DeathPlot = 1 - 1./(4.*(M+1)).*dot(1+tanh(k.*(x-y)),1+tanh(k.*(x-y)));
end
%% Parameter Test Solver
function [sol] = WaresDistributedImmunityResistantSolver(totaltime,IC,PA) %DDE model without therapy
% opts = ddeset('RelTol',1e-6,'AbsTol',1e-6,'MaxStep',1e-2,'Events',@EventsViralMonths1);
opts = odeset('RelTol',1e-8,'AbsTol',1e-8,'MaxStep',1e-2,'Events',@EventsViralMonths1);
sol = ode15s(@ViralOncologyParameterFit,totaltime,IC,opts);
%opts = ddeset('RelTol',1e-6,'AbsTol',1e-6,'MaxStep',1e-2,'Events',@EventsViralMonths1);
%sol = ddesd_f5(@Wares2012DistributedImmunity,@(t,y) DelayWares1(t,y,PA) ,IC,totaltime,opts);
function dydt = ViralOncologyParameterFit(t,y,Z);
%Quiescent cells
dydt(1) = 2.*(1-PA.nu).*PA.TransitRate.*y(PA.N+4)-(PA.a1+PA.d1+ psiQ(y(PA.N+6),y(1),PA) ).*y(1); 
%G1 Cells
dydt(2) = PA.a1.*y(1)-( PA.a2+PA.d2+ PA.kappa.*Infection(y(4),PA)+ psiS(y(PA.N+6),y(2),PA) ).*y(2); 
%Infected Cells
dydt(3) =  -PA.delta.*y(3)+ PA.kappa.*Infection(y(4),PA).*(y(2)+y(PA.N+7)+y(PA.N+9)+y(PA.N+PA.N+10) ); % PA.kappa.*Infection(y(1),y(2)+Sbar,y(3),y(4),PA).*(y(2)+Sbar); % Infected cells
%Virions
dydt(4) =   ViralDose(PA,t) + PA.alpha.*PA.delta.*y(3)-PA.omega.*y(4)- PA.kappa.*Infection(y(4),PA).*(y(2)+y(PA.N+7)+y(PA.N+9)+y(PA.N+PA.N+10)); % PA.kappa.*Infection(y(1),y(2)+Sbar,y(3),y(4),PA ).*(y(2)+Sbar); % Virions
%Writing ODE for first transit compartment
 dydt(5) = PA.a2.*y(2) - PA.TransitRate.*y(5) - ( PA.d3Hat+PA.kappa.*Infection(y(4),PA) + psiS(y(PA.N+6),y(5),PA) ).*y(5);
for jj = 6:PA.N+4
    dydt(jj) = PA.TransitRate.*(y(jj-1)-y(jj)) - ( PA.d3Hat +PA.kappa.*Infection(y(4),PA) + psiS(y(PA.N+6),y(jj),PA) ).*y(jj); %Transit compartment ODEs
end
%Immune cytokine 
dydt(PA.N+5) =   CProd( psiS(y(PA.N+6),y(2),PA).*y(2)+PA.delta.*y(3)+psiQ(y(PA.N+6),y(1),PA).*y(1),PA) - PA.kel.*y(PA.N+5)+Dose(PA,t); 
%Phagocytes
dydt(PA.N+6) =   PA.Kcp.*y(PA.N+5)./(PA.P12+y(PA.N+5)) - PA.gammaP.*y(PA.N+6); 
%ODE for total number of cells in cell cycle
dydt(PA.N+7) = PA.a2.*y(2) - (PA.d3Hat+PA.kappa.*Infection(y(4),PA)+psiS(y(PA.N+6),y(2),PA) ).*y(PA.N+7) - (PA.TransitRate./PA.a2).*y(PA.N+4);
% Resistant compartments
%Resistant Quiescent
dydt(PA.N+8) =   2.*PA.nu.*PA.TransitRate.*y(PA.N+4) + 2.*PA.TransitRate.*y(PA.N+PA.N+9)-(PA.a1R+PA.d1R).*y(PA.N+8); %Resistant quiescence DDE
%Resistant G1
dydt(PA.N+9) =   PA.a1R.*y(PA.N+8)-(PA.a2R+PA.d2R+ PA.kappa.*Infection(y(4),PA) ).*y(PA.N+9); %Susceptible resistant cells
%Resistant First transit
dydt(PA.N+10) =  PA.a2R.*y(PA.N+9) - PA.TransitRate.*y(PA.N+10) - (PA.d3HatR+PA.kappa.*Infection(y(4),PA)).*y(PA.N+10); %Susceptible resistant first compartment
for jj = PA.N+11:PA.N+PA.N+9
    dydt(jj) =  PA.TransitRate.*(y(jj-1)-y(jj)) - (PA.d3HatR +PA.kappa.*Infection(y(4),PA)  ).*y(jj); %Resistant Transit compartment ODEs
end
%DE for total resistant cells
dydt(PA.N+PA.N+10) =   PA.a2.*y(PA.N+9) - (PA.d3Hat+PA.kappa.*Infection(y(4),PA) ).*y(PA.N+PA.N+10) - (PA.TransitRate./PA.a2).*y(PA.N+PA.N+9);
dydt = dydt';
end
function [value,isterminal,direction] = EventsViralMonths1(t,y,Z)

value(1) = y(1)+ y(2) +y(PA.N+7)  - 2.*PA.TotalCells;  %What we are setting to 0, the tumour doubling size (different initial conditions, but close enough

isterminal(1) = 0;   % 1 = End the integration
direction(1) = 0;   % Positive direction only

value(2)= y(1)+ y(2)+ y(PA.N+7) - PA.DeathTime;  %What we are setting to 0 
isterminal(2) = 1;   % 1 = End the integration
direction(2) = 0;   % Positive direction only

end
end

%% Distributed Immunity  Solver
% function [sol] = WaresDistributedImmunityResistantSolver(totaltime,IC,PA) %DDE model without therapy
% opts = ddeset('RelTol',1e-6,'AbsTol',1e-6,'MaxStep',1e-2,'Events',@EventsViralMonths1);
% % opts = ddeset('RelTol',1e-6,'AbsTol',1e-6,'MaxStep',1e-2);
% % opts = ddeset('RelTol',1e-6,'AbsTol',1e-6,'MaxStep',1e-2); 
% sol = ddesd_f5(@Wares2012DistributedImmunity,@(t,y) DelayWares1(t,y,PA) ,IC,totaltime,opts);
% function dydt = Wares2012DistributedImmunity(t,y,Z);
% ylag1 = Z(:,1);
% Sbar = (1./PA.TransitRate).*ones(1,PA.N)*y(5:PA.N+4);
% RSbar = (1./PA.TransitRate).*ones(1,PA.N)*y(PA.N+9:PA.N+8+PA.N);
% dydt = zeros(PA.N+4+2,1); % Vector for RHS of ODE
% % ODEs for Distributed Delay from Crivelli 2012
% dydt(1) = 2.*(1-PA.nu).*PA.TransitRate.*y(PA.N+4)-(PA.a1+PA.d1+ psiQ(y(PA.N+6),y(1),PA) ).*y(1); %Quiescent cells
% dydt(2) = PA.a1.*y(1)-(PA.a2+PA.d2+ PA.kappa.*Infection(y(1),y(2)+Sbar,y(3),y(4),PA) + psiS(y(PA.N+6),y(2),PA)).*y(2); %Susceptible cells
% dydt(3) =  -PA.delta.*y(3)+PA.kappa.*Infection(y(1),y(2)+Sbar,y(3),y(4),PA).*(y(2)+Sbar+RSbar+y(PA.N+8)); % Infected cells
% dydt(4) =  ViralDose(PA,t) + PA.alpha.*y(3)-PA.omega.*y(4)-PA.kappa.*Infection(y(1),y(2)+Sbar,y(3),y(4),PA ).*(y(2)+Sbar); % Virions
% %Writing ODE for first transit compartment
% dydt(5) = PA.a2.*y(2) - PA.TransitRate.*y(5) - (PA.d3Hat+PA.kappa.*Infection(y(1),y(2)+Sbar,y(3),y(4),PA)+ psiS(y(PA.N+6),y(5),PA)).*y(5);
% for jj = 6:PA.N+4
%     dydt(jj) = PA.TransitRate.*(y(jj-1)-y(jj)) - (PA.d3Hat +PA.kappa.*Infection(y(1),y(2)+Sbar,y(3),y(4),PA) + psiS(y(PA.N+6),y(jj),PA)).*y(jj); %Transit compartment ODEs
% end
% dydt(PA.N+5) =  CProd( psiS(y(PA.N+6),y(2),PA).*y(2)+PA.delta.*y(3)+psiQ(y(PA.N+6),y(1),PA).*y(1),PA) - PA.kel.*y(PA.N+5)+Dose(PA,t); %Immune cytokine 
% dydt(PA.N+6) =  PA.Kcp.*y(PA.N+5)./(PA.P12+y(PA.N+5)) - PA.gammaP.*y(PA.N+6); %Phagocytes
% % Resistant compartments
% dydt(PA.N+7) = 2.*PA.nu.*PA.TransitRate.*y(PA.N+4) + 2.*PA.TransitRate.*y(PA.N+4+10)-(PA.a1R+PA.d1R).*y(PA.N+7); %Resistant quiescence DDE
% dydt(PA.N+8) = PA.a1R.*y(PA.N+7)-(PA.a2R+PA.d2R+ PA.kappa.*Infection(y(1),y(2)+Sbar,y(3),y(4),PA) ).*y(PA.N+8); %Susceptible resistant cells
% dydt(PA.N+9) = PA.a2R.*y(PA.N+8) - PA.TransitRate.*y(PA.N+9) - (PA.d3HatR+PA.kappa.*Infection(y(1),y(2)+Sbar,y(3),y(4),PA)).*y(PA.N+9); %Susceptible resistant first compartment
% for jj = PA.N+10:PA.N+PA.N+8
%     dydt(jj) = PA.TransitRate.*(y(jj-1)-y(jj)) - (PA.d3HatR +PA.kappa.*Infection(y(1),y(2)+Sbar,y(3),y(4),PA) ).*y(jj); %Resistant Transit compartment ODEs
% end
% end
% function [value,isterminal,direction] = EventsViralMonths1(t,y,Z)
% 
% value(1) = y(1)+ y(2) + y(PA.N+7)+y(PA.N+8) - 400;  %What we are setting to 0, the tumour doubling size (different initial conditions, but close enough
% 
% isterminal(1) = 0;   % 1 = End the integration
% direction(1) = 0;   % Positive direction only
% 
% value(2)= y(1)+ y(2)+ y(PA.N+7)+y(PA.N+8) - 400.*2^(13+PA.DeathTime);  %What we are setting to 0 
% isterminal(2) = 1;   % 1 = End the integration
% direction(2) = 0;   % Positive direction only
% 
% end
% end

function g = psiQ(P,Q,PA)
% clearance of quiescent cells by immune system
 g = P.*(PA.kp./(1+PA.kq.*Q));
% g = PA.kp.*P;
end

function h = psiS(P,S,PA)
% clearance of susceptible cells by immune system
h =  P.*(PA.kp./(1+PA.ks.*S));
% h = P.*PA.kp;
end

function y = CProd(a,PA)
%Cytokine Production
y = PA.CprodHomeo+ (PA.CprodMax-PA.CprodHomeo).*(a./(PA.C12+a));
end

function f = Infection(V,PA) %Contact function
    if V > 1e-10
        f = V./(PA.eta12+V);
    else
        f = 0;
    end
end

% Immunotherapy dosing function
function DoseWares = Dose(PA,t);
DoseVec = zeros(1,PA.AdminNumber);
TAdmin = PA.StartTime+(0:PA.AdminNumber-1).*PA.Offset-t;
for nn = 1:PA.AdminNumber
    if TAdmin(nn) < 0
    DoseVec(nn) = (PA.AvailFrac.*PA.kabs.*PA.Admin(nn))./PA.Vol.*exp(PA.kabs.*TAdmin(nn));
    else
        DoseVec(nn) = 0;
    end
end
DoseWares = ones(1,PA.AdminNumber)*DoseVec';
end

% Viral dosing function
function DoseViral = ViralDose(PA,t);
ViralDoseVec = zeros(1,PA.ViralAdminNumber);
TAdminViral = PA.ViralStartTime+(0:PA.ViralAdminNumber-1).*PA.ViralOffset-t;
for nn = 1:PA.ViralAdminNumber
    if TAdminViral(nn) < 0
    ViralDoseVec(nn) = (PA.Viralkabs.*PA.ViralAdmin(nn).*PA.ViralAvailFrac)./PA.Vol.*exp(PA.Viralkabs.*TAdminViral(nn));
    else
        ViralDoseVec(nn) = 0;
    end
end
DoseViral = ones(1,PA.ViralAdminNumber)*ViralDoseVec';
end

function d = DelayWares1(t,y,PA)
%This function sets up the delay vectors necessary for the DDE solver.
d = [t-PA.tau];            
end