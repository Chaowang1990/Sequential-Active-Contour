function phi =  TwoPhLSE(phi,d,timestep,epsilon)

phi=NeumannBoundCond(phi);
DiracF = Dirac(phi,epsilon);
%--------------------------
F = d(:,:,2)-d(:,:,1);
phi = phi + timestep*(F.*DiracF);
%------------------------Li
% K=curvature_central(phi);  
% F = d(:,:,2)-d(:,:,1);
% phi = phi + timestep*(0.02*(4*del2(phi)-K)+(F+0.00001*255^2*K).*DiracF);