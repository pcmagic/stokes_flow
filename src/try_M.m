b = 1; 
k = 1; 
Ra = 1;
rx = [1, 1, 1]';
X = [2, 2, 2]'; 
Xi = [3, 3, 3]';

disx = rx(1,b) - X(1,k);
disy = rx(2,b) - X(2,k);
disz = rx(3,b) - X(3,k);
r2sk = disx^2+disy^2+disz^2;
rsk = r2sk^0.5;
r3sk = r2sk^1.5;

H1sk = 1.0 / (r3sk);
MSxx(b,k) = ((r2sk)+disx^2)*H1sk;
MSyx(b,k) = disy*disx*H1sk;
MSxy(b,k) = MSyx(b,k);
MSyy(b,k) = ((r2sk)+disy^2)*H1sk;
MSxz(b,k) = disx*disz*H1sk;
MSyz(b,k) = disy*disz*H1sk;
MSzx(b,k) = MSxz(b,k);
MSzy(b,k) = MSyz(b,k);
MSzz(b,k) = ((r2sk)+disz^2)*H1sk;

disxi = rx(1,b) - Xi(1,k);
disyi = rx(2,b) - Xi(2,k);
diszi = rx(3,b) - Xi(3,k);
r2ski = disxi^2+disyi^2+diszi^2;
rski = r2ski^0.5;
r3ski = r2ski^1.5;
r5ski = r2ski^2.5;
H1ski = 1/rski;
H2ski = 1/r2ski;
H3ski = 1/r3ski;
H5ski = 1/r5ski;

Xf = norm(X(:,k));
Xfi= norm(Xi(:,k));
xf = norm(rx(:,b));
Dx1=dot(rx(:,b)-Xi(:,k),Xi(:,k));
Dx2=dot(rx(:,b),Xi(:,k));

A = 0.5*(Xf^2-Ra^2)/Xf^3;
B = r2ski*(rski-Xfi)*Xfi;
C = 3*Ra/Xfi;
Det = 1/(Xfi*(Xfi*rski+Dx2-Xfi^2));
E = 1/(xf*Xfi*(xf*Xfi+Dx2));

A1 = (Xf^2-Ra^2)/Xf;
A2 = xf^2-Ra^2;

Pxx(b,k) = A*(-3*X(1,k)*disxi*H3ski/Ra+Ra*H3ski-3*Ra*disxi^2*H5ski-2*X(1,k)*Xi(1,k)*H3ski/Ra+6*X(1,k)*H5ski*disxi*Dx1/Ra   +   C*(Xi(1,k)*disxi*r2ski+disxi^2*Xfi^2+B)*H3ski*Det   -   C*H2ski*Det^2*Xfi*(Xfi*disxi+Xi(1,k)*rski)*(Xi(1,k)*r2ski-disxi*Xfi^2+(rx(1,b)-2*Xi(1,k))*rski*Xfi)   -   C*E*(rx(1,b)*Xi(1,k)+xf*Xfi)   +   C*E^2*(Xfi*rx(1,b)+xf*Xi(1,k))*(Xfi*rx(1,b)+xf*Xi(1,k))*xf*Xfi );
Pyy(b,k) = A*(-3*X(2,k)*disyi*H3ski/Ra+Ra*H3ski-3*Ra*disyi^2*H5ski-2*X(2,k)*Xi(2,k)*H3ski/Ra+6*X(2,k)*H5ski*disyi*Dx1/Ra   +   C*(Xi(2,k)*disyi*r2ski+disyi^2*Xfi^2+B)*H3ski*Det   -   C*H2ski*Det^2*Xfi*(Xfi*disyi+Xi(2,k)*rski)*(Xi(2,k)*r2ski-disyi*Xfi^2+(rx(2,b)-2*Xi(2,k))*rski*Xfi)   -   C*E*(rx(2,b)*Xi(2,k)+xf*Xfi)   +   C*E^2*(Xfi*rx(2,b)+xf*Xi(2,k))*(Xfi*rx(2,b)+xf*Xi(2,k))*xf*Xfi );
Pzz(b,k) = A*(-3*X(3,k)*diszi*H3ski/Ra+Ra*H3ski-3*Ra*diszi^2*H5ski-2*X(3,k)*Xi(3,k)*H3ski/Ra+6*X(3,k)*H5ski*diszi*Dx1/Ra   +   C*(Xi(3,k)*diszi*r2ski+diszi^2*Xfi^2+B)*H3ski*Det   -   C*H2ski*Det^2*Xfi*(Xfi*diszi+Xi(3,k)*rski)*(Xi(3,k)*r2ski-diszi*Xfi^2+(rx(3,b)-2*Xi(3,k))*rski*Xfi)   -   C*E*(rx(3,b)*Xi(3,k)+xf*Xfi)   +   C*E^2*(Xfi*rx(3,b)+xf*Xi(3,k))*(Xfi*rx(3,b)+xf*Xi(3,k))*xf*Xfi );

Pxy(b,k) = A*(-3*X(2,k)*disxi*H3ski/Ra-3*Ra*disxi*disyi*H5ski-2*X(2,k)*Xi(1,k)*H3ski/Ra+6*X(2,k)*H5ski*disxi*Dx1/Ra+C*(Xi(2,k)*disxi*r2ski+disxi*disyi*Xfi^2)*H3ski*Det-C*H2ski*Det^2*Xfi*(Xfi*disxi+Xi(1,k)*rski)*(Xi(2,k)*r2ski-disyi*Xfi^2+(rx(2,b)-2*Xi(2,k))*rski*Xfi)-C*E*(rx(1,b)*Xi(2,k))+C*E^2*(Xfi*rx(1,b)+xf*Xi(1,k))*(Xfi*rx(2,b)+xf*Xi(2,k))*xf*Xfi);
Pyx(b,k) = A*(-3*X(1,k)*disyi*H3ski/Ra-3*Ra*disxi*disyi*H5ski-2*X(2,k)*Xi(1,k)*H3ski/Ra+6*X(1,k)*H5ski*disyi*Dx1/Ra+C*(Xi(1,k)*disyi*r2ski+disxi*disyi*Xfi^2)*H3ski*Det-C*H2ski*Det^2*Xfi*(Xfi*disyi+Xi(2,k)*rski)*(Xi(1,k)*r2ski-disxi*Xfi^2+(rx(1,b)-2*Xi(1,k))*rski*Xfi)-C*E*(rx(2,b)*Xi(1,k))+C*E^2*(Xfi*rx(1,b)+xf*Xi(1,k))*(Xfi*rx(2,b)+xf*Xi(2,k))*xf*Xfi);

Pxz(b,k) = A*(-3*X(3,k)*disxi*H3ski/Ra-3*Ra*disxi*diszi*H5ski-2*X(3,k)*Xi(1,k)*H3ski/Ra+6*X(3,k)*H5ski*disxi*Dx1/Ra+C*(Xi(3,k)*disxi*r2ski+disxi*diszi*Xfi^2)*H3ski*Det-C*H2ski*Det^2*Xfi*(Xfi*disxi+Xi(1,k)*rski)*(Xi(3,k)*r2ski-diszi*Xfi^2+(rx(3,b)-2*Xi(3,k))*rski*Xfi)-C*E*(rx(1,b)*Xi(3,k))+C*E^2*(Xfi*rx(1,b)+xf*Xi(1,k))*(Xfi*rx(3,b)+xf*Xi(3,k))*xf*Xfi);
Pzx(b,k) = A*(-3*X(1,k)*diszi*H3ski/Ra-3*Ra*disxi*diszi*H5ski-2*X(3,k)*Xi(1,k)*H3ski/Ra+6*X(1,k)*H5ski*diszi*Dx1/Ra+C*(Xi(1,k)*diszi*r2ski+disxi*diszi*Xfi^2)*H3ski*Det-C*H2ski*Det^2*Xfi*(Xfi*diszi+Xi(3,k)*rski)*(Xi(1,k)*r2ski-disxi*Xfi^2+(rx(1,b)-2*Xi(1,k))*rski*Xfi)-C*E*(rx(3,b)*Xi(1,k))+C*E^2*(Xfi*rx(1,b)+xf*Xi(1,k))*(Xfi*rx(3,b)+xf*Xi(3,k))*xf*Xfi);

Pyz(b,k) = A*(-3*X(3,k)*disyi*H3ski/Ra-3*Ra*diszi*disyi*H5ski-2*X(2,k)*Xi(3,k)*H3ski/Ra+6*X(3,k)*H5ski*disyi*Dx1/Ra+C*(Xi(3,k)*disyi*r2ski+diszi*disyi*Xfi^2)*H3ski*Det-C*H2ski*Det^2*Xfi*(Xfi*disyi+Xi(2,k)*rski)*(Xi(3,k)*r2ski-diszi*Xfi^2+(rx(3,b)-2*Xi(3,k))*rski*Xfi)-C*E*(rx(2,b)*Xi(3,k))+C*E^2*(Xfi*rx(3,b)+xf*Xi(3,k))*(Xfi*rx(2,b)+xf*Xi(2,k))*xf*Xfi);
Pzy(b,k) = A*(-3*X(2,k)*diszi*H3ski/Ra-3*Ra*disyi*diszi*H5ski-2*X(3,k)*Xi(2,k)*H3ski/Ra+6*X(2,k)*H5ski*diszi*Dx1/Ra+C*(Xi(2,k)*diszi*r2ski+disyi*diszi*Xfi^2)*H3ski*Det-C*H2ski*Det^2*Xfi*(Xfi*diszi+Xi(3,k)*rski)*(Xi(2,k)*r2ski-disyi*Xfi^2+(rx(2,b)-2*Xi(2,k))*rski*Xfi)-C*E*(rx(3,b)*Xi(2,k))+C*E^2*(Xfi*rx(2,b)+xf*Xi(2,k))*(Xfi*rx(3,b)+xf*Xi(3,k))*xf*Xfi);

Mxx(b,k) = MSxx(b,k)-Ra*H1ski/Xf-Ra^3*disxi*disxi*H3ski/Xf^3-A1*(Xi(1,k)*Xi(1,k)*H1ski/Ra^3-Ra*H3ski*(Xi(1,k)*disxi+Xi(1,k)*disxi)/Xf^2+2*Xi(1,k)*Xi(1,k)*Dx1*H3ski/Ra^3)-A2*Pxx(b,k);
Myy(b,k) = MSyy(b,k)-Ra*H1ski/Xf-Ra^3*disyi*disyi*H3ski/Xf^3-A1*(Xi(2,k)*Xi(2,k)*H1ski/Ra^3-Ra*H3ski*(Xi(2,k)*disyi+Xi(2,k)*disyi)/Xf^2+2*Xi(2,k)*Xi(2,k)*Dx1*H3ski/Ra^3)-A2*Pyy(b,k);
Mzz(b,k) = MSzz(b,k)-Ra*H1ski/Xf-Ra^3*diszi*diszi*H3ski/Xf^3-A1*(Xi(3,k)*Xi(3,k)*H1ski/Ra^3-Ra*H3ski*(Xi(3,k)*diszi+Xi(3,k)*diszi)/Xf^2+2*Xi(3,k)*Xi(3,k)*Dx1*H3ski/Ra^3)-A2*Pzz(b,k);

Mxy(b,k) = MSxy(b,k)-Ra^3*disxi*disyi*H3ski/Xf^3-A1*(Xi(1,k)*Xi(2,k)*H1ski/Ra^3-Ra*H3ski*(Xi(1,k)*disyi+Xi(2,k)*disxi)/Xf^2+2*Xi(2,k)*Xi(1,k)*Dx1*H3ski/Ra^3)-A2*Pxy(b,k);
Mxz(b,k) = MSxz(b,k)-Ra^3*disxi*diszi*H3ski/Xf^3-A1*(Xi(1,k)*Xi(3,k)*H1ski/Ra^3-Ra*H3ski*(Xi(1,k)*diszi+Xi(3,k)*disxi)/Xf^2+2*Xi(3,k)*Xi(1,k)*Dx1*H3ski/Ra^3)-A2*Pxz(b,k);
Myz(b,k) = MSyz(b,k)-Ra^3*disyi*diszi*H3ski/Xf^3-A1*(Xi(3,k)*Xi(2,k)*H1ski/Ra^3-Ra*H3ski*(Xi(3,k)*disyi+Xi(2,k)*diszi)/Xf^2+2*Xi(2,k)*Xi(3,k)*Dx1*H3ski/Ra^3)-A2*Pyz(b,k);

Myx(b,k) = MSyx(b,k)-Ra^3*disxi*disyi*H3ski/Xf^3-A1*(Xi(1,k)*Xi(2,k)*H1ski/Ra^3-Ra*H3ski*(Xi(1,k)*disyi+Xi(2,k)*disxi)/Xf^2+2*Xi(2,k)*Xi(1,k)*Dx1*H3ski/Ra^3)-A2*Pyx(b,k);
Mzx(b,k) = MSzx(b,k)-Ra^3*disxi*diszi*H3ski/Xf^3-A1*(Xi(1,k)*Xi(3,k)*H1ski/Ra^3-Ra*H3ski*(Xi(1,k)*diszi+Xi(3,k)*disxi)/Xf^2+2*Xi(3,k)*Xi(1,k)*Dx1*H3ski/Ra^3)-A2*Pzx(b,k);
Mzy(b,k) = MSzy(b,k)-Ra^3*disyi*diszi*H3ski/Xf^3-A1*(Xi(3,k)*Xi(2,k)*H1ski/Ra^3-Ra*H3ski*(Xi(3,k)*disyi+Xi(2,k)*diszi)/Xf^2+2*Xi(2,k)*Xi(3,k)*Dx1*H3ski/Ra^3)-A2*Pzy(b,k);

