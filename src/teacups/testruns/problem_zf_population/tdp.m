clear all

Sys.S =[0.5, 1];
Sys.g = [2.0059; 2.003];
Sys.D = [0 0; 700 0];

Sys.J = -20000;
%Sys.initState = {[0.3, 0.2, 0.2, 0.3, 0.5, 0.5], 'uncoupled'};
Sys.initState = {[0.1, 0.2, 0.3], 'from_triplet'};
Sys.pop_d = [0.2, 0.1];
Sys.lw = 1;

Exp.Range = [320, 380];
Exp.nPoints = 600;
Exp.mwFreq = 9.75;
Exp.Harmonic = 0;

sim = mypepper(Sys, Exp);
field = linspace(Exp.Range(1), Exp.Range(2), Exp.nPoints);
plot(field, sim)

A = [field; sim];
writematrix(A, "M.txt", "Delimiter", "tab")