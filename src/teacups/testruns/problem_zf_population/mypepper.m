function y = mypepper(Sys, Exp)
  if iscell(Sys.initState) && strcmp(Sys.initState{2},'from_triplet')
      pop = [Sys.initState{1}, Sys.pop_d];
      pop = pop/sum(pop)
      t = diag(pop(1: 3));
      syst.S = 1;
      syst.g = Sys.g(2);
      syst.D = [Sys.D(2), Sys.D(4)];
      [hamt] = ham(syst, [0 0 0]);
      [ZFStates, ZFEnergies] = eig(hamt);
      [~, idx] = sort(real(diag(ZFEnergies)));
      ZFStates = ZFStates(:, idx);
      t = ZFStates*t*ZFStates'


      d = diag(pop(4: end))
      c = kron(t, d)
      c = kron(eye(2), t) + kron(d, eye(3))
      %U2C = cgmatrix(1, 0.5);
      %coupled_states = U2C*c*U2C';
      Sys.initState = {c, 'uncoupled'};

  end
  
  [~, y] = pepper(Sys, Exp);
  y = normalize(y);
end