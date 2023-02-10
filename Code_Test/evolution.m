function [phi1,phi2] = evolution(phi1,d,epsilon,timestep,phi2)

    phi1=NeumannBoundCond(phi1);
    H1 = Heaviside(phi1,epsilon);
    Dirac1 = Dirac(phi1,epsilon);
 
    phi2=NeumannBoundCond(phi2);
    H2 = Heaviside(phi2,epsilon);
    Dirac2 = Dirac(phi2,epsilon);
    DataF1 = -(d(:,:,1)-d(:,:,2)-d(:,:,3)+d(:,:,4)).*H2-(d(:,:,2)-d(:,:,4));

    phi1 = phi1 +timestep*(DataF1.*Dirac1); 
    DataF2 = -(d(:,:,1)-d(:,:,2)-d(:,:,3)+d(:,:,4)).*H1-(d(:,:,3)-d(:,:,4));
    phi2 = phi2 +timestep*(DataF2.*Dirac2);
