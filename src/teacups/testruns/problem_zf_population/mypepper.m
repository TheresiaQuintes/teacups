function y = mypepper(Sys, Exp)
  if iscell(Sys.initState) && strcmp(Sys.initState{2},'zerofield')
  
    zfpops = Sys.initState{1};

    % Get zero-field states and order in terms of energy
    [F,~,~,~] = ham(Sys);
    [ZFStates,ZFEnergies] = eig(F)
    [ZFEnergies,idx] = sort(real(diag(ZFEnergies)));
    ZFStates = ZFStates(:,idx);

    % Convert population vector to density matrix (in uncoupled basis)
    Sys.initState = ZFStates*diag(zfpops)*ZFStates';

  end
  
  if iscell(Sys.initState) && strcmp(Sys.initState{2},'from_triplet')
  
      t = diag(Sys.initState{1});
      syst.S = 1;
      syst.g = Sys.g(1);
      syst.D = [Sys.D(1), Sys.D(3)];
      hamt = ham(syst, [0, 0, 0])
      [ZFStates, ZFEnergies] = eig(hamt);
      [ZFEnergies, idx] = sort(real(diag(ZFEnergies)));
      ZFStates = ZFStates(:, idx);
      t = ZFStates*t*ZFStates'


      d = diag([Sys.pop_d])
      c = kron(t, eye(2)) + kron(eye(3), d)
      %c = kron(t, d)
      Sys.initState = {c, 'uncoupled'};
      %U2C = cgmatrix(1, 0.5);
      %coupled_states = U2C*c*U2C';
      %Sys.initState = {coupled_states, 'coupled'};

  end
  [~, y] = pepper(Sys, Exp);
  y = y/max(abs(y));
end