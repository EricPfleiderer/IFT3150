##% Fitness function
#function [Obj] = Updated16052019TreatmentOptimizerObjective(x,PA)   
## Time interval
#tf = 3; #Simulation over 2.5 months of treatment + 0.5 after treatment
#tstep = floor(tf./20);
#totaltime = [0 tf];
#
##Load the dosing strategy for immunotherapy
#ImmunoTherapyDoses = x(1:PA.AdminNumber);
#
##Immunotherapy
#PA.StartTime = 0;
#PA.Offset = 1./30; #Dosed every day (some Admin can be 0)
#PA.Admin = 1.*125000.*ImmunoTherapyDoses; # %Amount of immunotherapy administered 
#PA.Vol = 7; #volume of absorption
#PA.kabs = 6.6311.*30; # absorption rate
#PA.AvailFrac =0.85; 
#
##Load the dosing strategy for the viral therapy
#ViralTherapyDoses = x(PA.AdminNumber+1:end);
#
## Viral therapy
#PA.ViralStartTime = 0;
#PA.ViralOffset = 7./30; #Dosed at a maximum of once weekly
##Load the dosing strategy for the viral therapy
#PA.ViralAdmin =  250.*ViralTherapyDoses; #Amount of virus administered (input from the optimizer)
#PA.Viralkabs = 20.*30; # absorption rate
#PA.ViralAvailFrac = 1;
#
##Calculate the total dose given:
#ViralCytokineConversion = 1; #Convert between immunotherapy and viral treatment load (If we think viral therapy is more/less burdensome for patients)
#TotalDose = dot(PA.ViralAdmin,ones(1,PA.ViralAdminNumber)) + ViralCytokineConversion.*dot(PA.Admin,ones(1,PA.AdminNumber)); 

##% Calculate the initial conditions
## Calculate the cell cycle duration
def init_cond(PA):
    """
    :param PA: Params class
    """
	TotalTime = 1/PA.a1 +1/(PA.a2+PA.d2)+PA.tau;
	PA.TotalCells = 200;
	# Initial Conditions
	QIC = (1/PA.a1 / TotalTime) * PA.TotalCells * (1 - PA.nu); #100;
	SIC = (1/(PA.a2 + PA.d2) / TotalTime) * PA.TotalCells * (1 - PA.nu);  #100;
	TCIC = (PA.tau / TotalTime) * PA.TotalCells * (1 - PA.nu)* ones(1, PA.N) / PA.N; #Transit compartment ICs
	NCIC = (PA.tau / TotalTime) * PA.TotalCells * (1 - PA.nu);
	IIC = 0;
	VIC = 0;
	CIC =  PA.CprodHomeo / PA.kel;
	PIC =   PA.Kcp * CIC / ((PA.P12 + CIC) * PA.gammaP);
	# Resistant Strain ICs
	RIC =   (1/PA.a1 / TotalTime) * PA.TotalCells * (PA.nu); #
	RSIC =   (1/(PA.a2 + PA.d2) / TotalTime) * PA.TotalCells * (PA.nu); #
	ResistantTCIC =   (PA.tau / TotalTime) * PA.TotalCells * (PA.nu) * ones(1, PA.N) / PA.N; 
	ResistantTotalCellsIC = (PA.tau / TotalTime) * PA.TotalCells * (PA.nu);
	InitialConditions = [QIC, SIC, IIC, VIC, TCIC, CIC, PIC, NCIC, RIC, RSIC, ResistantTCIC, ResistantTotalCellsIC];
        return InitialConditions
        
## Solve the distributed DDE
#[solTreat] = WaresDistributedImmunityResistantSolver(totaltime,InitialConditions,PA); #Calculate death time with viral treatment
#
##Objective function: Area under the tumour curve+area under treatment curve
#TimeSeries = linspace(0,tf,1001); #Create 1000 evenly space points inside the time domain
#IntStep = TimeSeries(2)-TimeSeries(1); #Calculate the time step between points
#EvalSol = deval(solTreat,TimeSeries); #Evaluate the solution at the collocation points
#TrapFun = EvalSol(1,:)+ EvalSol(2,:); # Find the tumour burden at the collocation points.
#NTime = length(TimeSeries);
##Calculate the area under the tumour curve using Composite Simpson's rule
#TimeInt = 2.*ones(1,NTime);
#for ii = 1:500
#    TimeInt(2.*ii) = 4;
#end
#TimeInt(1) = 1;
#TimeInt(NTime) = 1;
#CumulativeTumourBurden = (IntStep/3).*(dot(TimeInt,TrapFun)); #Calculate the tumour AOC .
##The objective function we are trying to minimize.
#Obj = CumulativeTumourBurden + TotalDose;
#
##% Parameter Test Solver
#
#function [sol] = WaresDistributedImmunityResistantSolver(totaltime,IC,PA) #DDE model without therapy
## opts = ddeset('RelTol',1e-6,'AbsTol',1e-6,'MaxStep',1e-2,'Events',@EventsViralMonths1);
#opts = odeset('RelTol',1e-8,'AbsTol',1e-8,'MaxStep',1e-2,'Events',@EventsViralMonths1);
#sol = ode15s(@ViralOncologyParameterFit,totaltime,IC,opts);
##opts = ddeset('RelTol',1e-6,'AbsTol',1e-6,'MaxStep',1e-2,'Events',@EventsViralMonths1);
##sol = ddesd_f5(@Wares2012DistributedImmunity,@(t,y) DelayWares1(t,y,PA) ,IC,totaltime,opts);
#function dydt = ViralOncologyParameterFit(t,y,Z);
## ylag1 = Z(:,1);
## Sbar = (PA.TransitRate./PA.a2).*y(PA.N+4); 
##Quiescent cells
#dydt(1) = 2.*(1-PA.nu).*PA.TransitRate.*y(PA.N+4)-(PA.a1+PA.d1+ psiQ(y(PA.N+6),y(1),PA) ).*y(1); 
##G1 Cells
#dydt(2) = PA.a1.*y(1)-( PA.a2+PA.d2+ PA.kappa.*Infection(y(4),PA)+ psiS(y(PA.N+6),y(2),PA) ).*y(2); 
##Infected Cells
#dydt(3) =  -PA.delta.*y(3)+ PA.kappa.*Infection(y(4),PA).*(y(2)+y(PA.N+7)+y(PA.N+9)+y(PA.N+PA.N+10) ); # PA.kappa.*Infection(y(1),y(2)+Sbar,y(3),y(4),PA).*(y(2)+Sbar); % Infected cells
##Virions
#dydt(4) =   ViralDose(PA,t) + PA.alpha.*PA.delta.*y(3)-PA.omega.*y(4)- PA.kappa.*Infection(y(4),PA).*(y(2)+y(PA.N+7)+y(PA.N+9)+y(PA.N+PA.N+10)); # PA.kappa.*Infection(y(1),y(2)+Sbar,y(3),y(4),PA ).*(y(2)+Sbar); % Virions
##Writing ODE for first transit compartment
# dydt(5) = PA.a2.*y(2) - PA.TransitRate.*y(5) - ( PA.d3Hat+PA.kappa.*Infection(y(4),PA) + psiS(y(PA.N+6),y(5),PA) ).*y(5);
#for jj = 6:PA.N+4
#    dydt(jj) = PA.TransitRate.*(y(jj-1)-y(jj)) - ( PA.d3Hat +PA.kappa.*Infection(y(4),PA) + psiS(y(PA.N+6),y(jj),PA) ).*y(jj); #Transit compartment ODEs
#end
##Immune cytokine 
#dydt(PA.N+5) =    CProd( psiS(y(PA.N+6),y(2),PA).*y(2)+PA.delta.*y(3)+psiQ(y(PA.N+6),y(1),PA).*y(1),PA) - PA.kel.*y(PA.N+5)+Dose(PA,t); 
##Phagocytes
#dydt(PA.N+6) =   PA.Kcp.*y(PA.N+5)./(PA.P12+y(PA.N+5)) - PA.gammaP.*y(PA.N+6); 
##ODE for total number of cells in cell cycle
#dydt(PA.N+7) = PA.a2.*y(2) - (PA.d3Hat+PA.kappa.*Infection(y(4),PA)+psiS(y(PA.N+6),y(2),PA) ).*y(PA.N+7) - (PA.TransitRate./PA.a2).*y(PA.N+4);
## Resistant compartments
##Resistant Quiescent
#dydt(PA.N+8) =   2.*PA.nu.*PA.TransitRate.*y(PA.N+4) + 2.*PA.TransitRate.*y(PA.N+PA.N+9)-(PA.a1R+PA.d1R).*y(PA.N+8); #Resistant quiescence DDE
##Resistant G1
#dydt(PA.N+9) =   PA.a1R.*y(PA.N+8)-(PA.a2R+PA.d2R+ PA.kappa.*Infection(y(4),PA) ).*y(PA.N+9); #Susceptible resistant cells
##Resistant First transit
#dydt(PA.N+10) =  PA.a2R.*y(PA.N+9) - PA.TransitRate.*y(PA.N+10) - (PA.d3HatR+PA.kappa.*Infection(y(4),PA)).*y(PA.N+10); #Susceptible resistant first compartment
#for jj = PA.N+11:PA.N+PA.N+9
#    dydt(jj) =  PA.TransitRate.*(y(jj-1)-y(jj)) - (PA.d3HatR +PA.kappa.*Infection(y(4),PA)  ).*y(jj); #Resistant Transit compartment ODEs
#end
##DE for total resistant cells
#dydt(PA.N+PA.N+10) =   PA.a2.*y(PA.N+9) - (PA.d3Hat+PA.kappa.*Infection(y(4),PA) ).*y(PA.N+PA.N+10) - (PA.TransitRate./PA.a2).*y(PA.N+PA.N+9);
#dydt = dydt';
#end
#function [value,isterminal,direction] = EventsViralMonths1(t,y,Z)
#
#value(1) = y(1)+ y(2) +y(PA.N+7)  - 2.*PA.TotalCells;  #What we are setting to 0, the tumour doubling size (different initial conditions, but close enough
#
#isterminal(1) = 0;   # 1 = End the integration
#direction(1) = 0;   # Positive direction only
#
#value(2)= y(1)+ y(2)+ y(PA.N+7) - 5e5; #PA.TotalCells.*2^(13+PA.DeathTime);  %What we are setting to 0 
#isterminal(2) = 1;   # 1 = End the integration
#direction(2) = 0;   # Positive direction only
#
#end
#end
#
#
#function g = psiQ(P,Q,PA)# clearance of quiescent cells by immune system
# g = P.*(PA.kp./(1+PA.kq.*Q));
#end
#
#function h = psiS(P,S,PA) # clearance of susceptible cells by immune system
#h =  P.*(PA.kp./(1+PA.ks.*S));
#end
#
#function y = CProd(a,PA)
##Cytokine Production
#y = PA.CprodHomeo+ (PA.CprodMax-PA.CprodHomeo).*(a./(PA.C12+a));
#end
#
#function f = Infection(V,PA) #Contact function
#    if V > 1e-10
#        f = V./(PA.eta12+V);
#    else
#        f = 0;
#    end
#end
#
## Immunotherapy dosing function
#function DoseWares = Dose(PA,t);
#DoseVec = zeros(1,PA.AdminNumber);
#TAdmin = PA.StartTime+(0:PA.AdminNumber-1).*PA.Offset-t;
#for nn = 1:PA.AdminNumber
#    if TAdmin(nn) < 0
#    DoseVec(nn) = (PA.AvailFrac.*PA.kabs.*PA.Admin(nn))./PA.Vol.*exp(PA.kabs.*TAdmin(nn));
#    else
#        DoseVec(nn) = 0;
#    end
#end
#DoseWares = ones(1,PA.AdminNumber)*DoseVec';
#end
#
## Viral dosing function
#function DoseViral = ViralDose(PA,t);
#ViralDoseVec = zeros(1,PA.ViralAdminNumber);
#TAdminViral = PA.ViralStartTime+(0:PA.ViralAdminNumber-1).*PA.ViralOffset-t;
#for nn = 1:PA.ViralAdminNumber
#    if TAdminViral(nn) < 0
#    ViralDoseVec(nn) = (PA.Viralkabs.*PA.ViralAdmin(nn).*PA.ViralAvailFrac)./PA.Vol.*exp(PA.Viralkabs.*TAdminViral(nn));
#    else
#        ViralDoseVec(nn) = 0;
#    end
#end
#DoseViral = ones(1,PA.ViralAdminNumber)*ViralDoseVec';
#end
#
#
#end
# 
