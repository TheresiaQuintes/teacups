clear all

Sys.S =[1, 0.5];
Sys.g = [2.003; 2.0059];
Sys.D = [700 -100; 0 0];

Sys.J = -400000;
Sys.initState = {[0.1, 0.1, 0.2], 'from_triplet'};
%Sys.initState = {[0.3, 0.1, 0.1, 0.3, 0.5, 0.55], 'eigen'};
%Sys.initState = {[0.3 0.3667 0.3667 0.3 0.3333 0.3333], 'coupled'};
Sys.pop_d = [0.21 0.2];
Sys.lw = 0.4;

Exp.Range = [320, 380];
Exp.nPoints = 700;
Exp.mwFreq = 9.75;
Exp.Harmonic = 0;


sim = mypepper(Sys, Exp);
sim = sim/max(abs(sim));
field = linspace(Exp.Range(1), Exp.Range(2), Exp.nPoints);
plot(field, sim)

A = [field; sim];
writematrix(A, "M.txt", "Delimiter", "tab")

% for J_ex = [0, -10, -50, -100, -200, -500, -1000, -2000, -3000, -4000, -5000, -5500, -5800, -6000, -6300, -6500, -7000, -7500, -8000, -10000, -15000, -20000]
%     Sys.J = J_ex;
%     sim = mypepper(Sys, Exp);
%     sim = sim/max(abs(sim));
%     field = linspace(Exp.Range(1), Exp.Range(2), Exp.nPoints);
%     figure()
%     plot(field, sim)
%     A = [field; sim];
%     writematrix(A, join(["sim_zf_",int2str(J_ex), ".txt"], ""), "Delimiter", "tab")
%     hold off
% end
