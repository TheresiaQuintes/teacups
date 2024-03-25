clear all
% Spin system
Sys.S = [1/2 1/2];
Sys.g = [2.0020; 1.9999];

Sys.J = -5;  % MHz
Sys.dip = 10;  % MHz

Sys.lwpp = 0.05;  % mT

% Singlet precursor
Sys.initState = 'singlet';
Sys.initState = {[0.15 0.7 0.14 0],'coupled'};

% Experimental parameters
Exp.Range = [346.5 349.5];  % mT
Exp.nPoints = 600;
Exp.mwFreq = 9.75;  % GHz
Exp.Harmonic = 0;  % no field modulation

sim = pepper(Sys,Exp);



field = linspace(Exp.Range(1), Exp.Range(2), Exp.nPoints);
sim = sim/max(abs(sim));
plot(field, sim)
A = [field; sim];
writematrix(A, "M_rp.txt", "Delimiter", "tab")
