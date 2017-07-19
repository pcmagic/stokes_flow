%% Force-free and torque-free for Gaussian blob

Nhead = 1000;
ppp   = 580;
shift = 0.98;
Ra 	  = 0.001*0.98;
circleN = 12;
np = 1;

dis   = 0.2;
locR1 = [0.0 0.0 0.0];
the0  = 0.0;

epsA = 0.0;

Ohead = 0;
Otail = -200;
bodyU=[0.0 0.0 0.0 0.0 0.0 0.0];
nt = ppp*circleN*np*2;



bac1  = OneEcoli(Nhead, ppp, np, circleN, Ra, dis, the0, epsA, shift);
vbac1 = BC_Ecoli(bac1, bodyU, Ohead, Otail, locR1);

Ra = 0.001;

bac2  = OneEcoli(Nhead, ppp, np, circleN, Ra, dis, the0, epsA, shift);
vbac2 = BC_Ecoli(bac2, bodyU, Ohead, Otail, locR1);


rn = size(vbac1.rxEcoli,2);
fn = size(vbac1.gxEcoli,2);

	Mxx = zeros(rn, fn);
	Mxy = zeros(rn, fn);
	Mxz = zeros(rn, fn);
	Myx = zeros(rn, fn);
	Myy = zeros(rn, fn);
	Myz = zeros(rn, fn);
	Mzx = zeros(rn, fn);
	Mzy = zeros(rn, fn);
	Mzz = zeros(rn, fn);

for b = 1:rn        
    for k = 1:fn    
        temp_disx = vbac2.rxEcoli(b) - vbac1.gxEcoli(k);
        temp_disy = vbac2.ryEcoli(b) - vbac1.gyEcoli(k);
        temp_disz = vbac2.rzEcoli(b) - vbac1.gzEcoli(k);
        
        r2sk = temp_disx^2 + temp_disy^2 + temp_disz^2; 
		
		
		rsk = (r2sk)^1.5;
        
        H1sk = 1.0 / (8.0*pi*rsk);
        
        Mxx(b,k) = ((r2sk)+temp_disx^2)*H1sk;
        Myx(b,k) = temp_disy*temp_disx*H1sk;
        Mxy(b,k) = Myx(b,k);
        Myy(b,k) = ((r2sk)+temp_disy^2)*H1sk;
        Mxz(b,k) = temp_disx*temp_disz*H1sk;
        Myz(b,k) = temp_disy*temp_disz*H1sk;
        Mzx(b,k) = Mxz(b,k);
        Mzy(b,k) = Myz(b,k);
        Mzz(b,k) = ((r2sk)+temp_disz^2)*H1sk;
		
    end
end

uM = [Mxx Mxy Mxz; Myx Myy Myz; Mzx Mzy Mzz];

clear Mxx Mxy Mxz Myx Myy Myz Mzx Mzy Mzz;

	Xh_com=[0 0 0];
   
    Mb(1,:)=[repmat([1 0 0],1,Nhead+nt)];
    Mb(2,:)=[repmat([0 1 0],1,Nhead+nt)];
    Mb(3,:)=[repmat([0 0 1],1,Nhead+nt)];
    
    temp1=[zeros(Nhead,1) -(bac1.gzH'-Xh_com(3)) (bac1.gyH'-Xh_com(2))]';
    temp2=[zeros(nt,1)    -(bac1.gzT'-Xh_com(3)) (bac1.gyT'-Xh_com(2))]';
	Mb(4,:)=[temp1(:)' temp2(:)'];
    
    temp1=[(bac1.gzH'-Xh_com(3)) zeros(Nhead,1) -(bac1.gxH'-Xh_com(1))]';
	temp2=[(bac1.gzT'-Xh_com(3)) zeros(nt,1) -(bac1.gxT'-Xh_com(1))]';
	Mb(5,:)=[temp1(:)' temp2(:)'];	
    temp1=[-(bac1.gyH'-Xh_com(2)) (bac1.gxH'-Xh_com(1)) zeros(Nhead,1)]';
	temp2=[-(bac1.gyT'-Xh_com(2)) (bac1.gxT'-Xh_com(1)) zeros(nt,1)]';
	Mb(6,:)=[temp1(:)' temp2(:)'];
    
    Mb=[Mb(:,1:3:3*(nt+Nhead)) Mb(:,2:3:3*(nt+Nhead)) Mb(:,3:3:3*(nt+Nhead))];
    
    
    Mr(:,1)=[repmat([-1 0 0]',Nhead+nt,1)];
	Mr(:,2)=[repmat([0 -1 0]',Nhead+nt,1)];
	Mr(:,3)=[repmat([0 0 -1]',Nhead+nt,1)];
	temp1=[zeros(Nhead,1) (bac2.rzH'-Xh_com(3)) -(bac2.ryH'-Xh_com(2))]';
	temp2=[zeros(nt,1)    (bac2.rzT'-Xh_com(3)) -(bac2.ryT'-Xh_com(2))]';
   
	Mr(:,4)=[temp1(:);temp2(:)];
    temp1=[-(bac2.rzH'-Xh_com(3)) zeros(Nhead,1) (bac2.rxH'-Xh_com(1))]';
	temp2=[-(bac2.rzT'-Xh_com(3)) zeros(nt,1)    (bac2.rxT'-Xh_com(1))]';
  
	Mr(:,5)=[temp1(:);temp2(:)];	
    temp1=[(bac2.ryH'-Xh_com(2)) -(bac2.rxH'-Xh_com(1)) zeros(Nhead,1)]';
	temp2=[(bac2.ryT'-Xh_com(2)) -(bac2.rxT'-Xh_com(1)) zeros(nt,1)]';                
   
	Mr(:,6)=[temp1(:);temp2(:)];	
    
    Mr=[Mr(1:3:3*(nt+Nhead),:);Mr(2:3:3*(nt+Nhead),:);Mr(3:3:3*(nt+Nhead),:);zeros(6,6)];

    uM=[uM;Mb];
    uM=[uM Mr];

    clear Mb Mr;
	
	U = [vbac2.uEcoli vbac2.vEcoli vbac2.wEcoli];
	U = [U(:);zeros(6,1)];
	
Force = gmres(uM,U,3*rn-50,1e-6);
clear uM U;


u = Force(end-5:end);
F = Force(1:end-6);
OmegaT = Otail+u(4);
clear Mxx Mxy Mxz Myx Myy Myz Mzx Mzy Mzz uM;

save checkdata;
%bodyU=[u(1) u(2) u(3) 0 u(5) u(6)];

%save data2;



%{


Nhead = 3000;

bac3  = OneEcoli3(Nhead, ppp, np, circleN, Ra, dis, the0, epsA, shift);
vbac3 = BC_Ecoli(bac3, bodyU, u(4), OmegaT, locR1);

rn3 = size(vbac3.rxEcoli,2);

	Mxx = zeros(rn3, fn);
	Mxy = zeros(rn3, fn);
	Mxz = zeros(rn3, fn);
	Myx = zeros(rn3, fn);
	Myy = zeros(rn3, fn);
	Myz = zeros(rn3, fn);
	Mzx = zeros(rn3, fn);
	Mzy = zeros(rn3, fn);
	Mzz = zeros(rn3, fn);

for b = 1:rn3        
    for k = 1:fn    
        temp_disx = vbac3.rxEcoli(b) - vbac1.gxEcoli(k);
        temp_disy = vbac3.ryEcoli(b) - vbac1.gyEcoli(k);
        temp_disz = vbac3.rzEcoli(b) - vbac1.gzEcoli(k);
        
        r2sk = temp_disx^2 + temp_disy^2 + temp_disz^2; 
		
		
		rsk = (r2sk)^1.5;
        
        H1sk = 1.0 / (8.0*pi*rsk);
        
        Mxx(b,k) = ((r2sk)+temp_disx^2)*H1sk;
        Myx(b,k) = temp_disy*temp_disx*H1sk;
        Mxy(b,k) = Myx(b,k);
        Myy(b,k) = ((r2sk)+temp_disy^2)*H1sk;
        Mxz(b,k) = temp_disx*temp_disz*H1sk;
        Myz(b,k) = temp_disy*temp_disz*H1sk;
        Mzx(b,k) = Mxz(b,k);
        Mzy(b,k) = Myz(b,k);
        Mzz(b,k) = ((r2sk)+temp_disz^2)*H1sk;
		
    end
end

uM = [Mxx Mxy Mxz; Myx Myy Myz; Mzx Mzy Mzz];

V2 = uM*F;

	vox = V2(1:rn3);
	voy = V2(rn3+1:2*rn3);
	voz = V2(2*rn3+1:3*rn3);
	voxH=vox(1:Nhead);
	voyH=voy(1:Nhead);
	vozH=voz(1:Nhead);
	voxT=vox(Nhead+1:rn3);
	voyT=voy(Nhead+1:rn3);
	vozT=voz(Nhead+1:rn3);
	
	vxH = vbac3.uEcoli(1:Nhead)';
	vyH = vbac3.vEcoli(1:Nhead)';
	vzH = vbac3.wEcoli(1:Nhead)';

	vxT = vbac3.uEcoli(Nhead+1:rn3)';
	vyT = vbac3.vEcoli(Nhead+1:rn3)';
	vzT = vbac3.wEcoli(Nhead+1:rn3)';
	
	vT = (vxT.^2+vyT.^2+vzT.^2);
	criterionT = ((voxT-vxT).^2+(voyT-vyT).^2+(vozT-vzT).^2)/(mean(vT));
    
    vH = (vxH.^2+vyH.^2+vzH.^2);
    criterionH = ((voxH-vxH).^2+(voyH-vyH).^2+(vozH-vzH).^2)/mean(vH);
	CRIH = sqrt(mean(criterionH));
	CRIT = sqrt(mean(criterionT));
	disp(CRIT);
    disp(CRIH);
clear Mxx Mxy Mxz Myx Myy Myz Mzx Mzy Mzz uM;

%}