%% ------------------- 
% Use parts to assembly one Ecoli
% date: 2015-10-22

function bac_str = OneEcoli(Nhead,ppp,np,circleN,Ra,dis,the0,epsA,shift)

headA = 0.5;
headC = 0.5;

bHead = Spheroid(Nhead,headA,headC,shift);
rxH = bHead.rxS(1:Nhead) + headC;
ryH = bHead.ryS(1:Nhead);
rzH = bHead.rzS(1:Nhead);
gxH = bHead.gxS(1:Nhead) + headC;
gyH = bHead.gyS(1:Nhead);
gzH = bHead.gzS(1:Nhead);

% the helical tail
 Ntail = ppp*np*circleN;
% the0 = 0;

hTail1 = HelicalTail(ppp,np,circleN,Ra,the0);
rxtail1 = 0.0 - dis - hTail1.rxtail(1:Ntail);
rytail1 = hTail1.rytail(1:Ntail);
rztail1 = hTail1.rztail(1:Ntail);
gxtail1 = 0.0 - dis - hTail1.gxtail(1:Ntail);
gytail1 = hTail1.gytail(1:Ntail);
gztail1 = hTail1.gztail(1:Ntail);

hTail2 = HelicalTail(ppp,np,circleN,Ra,the0+pi);
rxtail2 = 0.0 - dis - hTail2.rxtail(1:Ntail);
rytail2 = hTail2.rytail(1:Ntail);
rztail2 = hTail2.rztail(1:Ntail);
gxtail2 = 0.0 - dis - hTail2.gxtail(1:Ntail);
gytail2 = hTail2.gytail(1:Ntail);
gztail2 = hTail2.gztail(1:Ntail);

rxT = [rxtail1 rxtail2];
ryT = [rytail1 rytail2];
rzT = [rztail1 rztail2];
gxT = [gxtail1 gxtail2];
gyT = [gytail1 gytail2];
gzT = [gztail1 gztail2];


for i = 1:Nhead
    for j = 1:Nhead
        if i==j
            dsh(i,j) = 10.0;
        else
        dsh(i,j)=abs(sqrt((gxH(j)-gxH(i))^2+(gyH(j)-gyH(i))^2+(gzH(j)-gzH(i))^2));
        end
    end
end
    if epsA==0
        epsH=epsA*min(dsh);
    else
        epsH=0.0*min(dsh);
    end
	
for i = 1:Ntail
    for j = 1:Ntail
        if i==j
            dst1(i,j) = 10.0;
        else
            dst1(i,j)=abs(sqrt((gxtail1(j)-gxtail1(i))^2+(gytail1(j)-gytail1(i))^2+(gztail1(j)-gztail1(i))^2));
        end
    end
end

for i = 1:Ntail
    for j = 1:Ntail
        if i==j
            dst2(i,j) = 10.0;
        else
            dst2(i,j)=abs(sqrt((gxtail2(j)-gxtail2(i))^2+(gytail2(j)-gytail2(i))^2+(gztail2(j)-gztail2(i))^2));
        end
    end
end

    epsT1=epsA*ones(Ntail,1)';
    epsT2=epsA*ones(Ntail,1)';
    %epsT1=epsA*min(dst1);
    %epsT2=epsA*min(dst2);
    eps=[epsH epsT1 epsT2];


bac_str = struct('rxH',rxH,'ryH',ryH,'rzH',rzH,'gxH',gxH,'gyH',gyH,'gzH',gzH,'rxT',rxT,'ryT',ryT,'rzT',rzT,'gxT',gxT,'gyT',gyT,'gzT',gzT,'eps',eps);

