% close all
clear all

fts=120;          % frequency of accelerometer aquisiton
tdiff=1.5085;     % t shift between accelerometer and dspace
nfilter=10;
nfilter2=10;
Resistances=['k30';'k20';'k15';'k10'];
Res2=[30e3;20e3;15e3;10e3];  % Ressistance values
Movement1=['p01';'p02';'p03';'p04';'p05';'p06';'p07';'p08';'p09';'p10']; %'p0[1-10]
Mass1=['LW';'HV'];   %'LW' 'HV'
Coils1=['4c';'2c'];  %'2c' '4c

Neuler=1;    %Euler force (yes =1, no =0)
Fcent=1;     % Centrifugal force (yes =1, no =0)

                       

for iResistace=1  %# of the resistance
    for iMovement=4
        for iMass=1
            for iCoils=1
                
                
                NumericalCalc=1;     % Numerically calculate output position, velocity and voltage from accelerometer data
                Animation=0;         % Animate trajectory
                
                MassMagnet=19.8e-3;
                MassWeight=0;
                
                nCorr=100;                       % Number of initial points where the system is still (to estimate gravity acceleration)
                L_E1_E=[-119.15;0*(-22+17.8);-27.85+10]*1e-3;   % [-119.15;0*(-22+17.8);-27.85+10]*1e-3; Vector between accelerometer referential's origin (basis E) and center of container (basis E1)   ([-119.15;17.8;-27.85])
                Xcm3=(0)*1e-3;      % Center of mass of the moving mass in E1
                
                Dampc=0.1;                    % c Drag constant in N/(m/s) (~0.25-0.4) (Drag force = -mass*Dampf*sign(v)-Dampc*v)1
                Dampf=0.0;                    % f/m Friction force in N/kg (~ 0.4*|sin(beta)| for Teflon)
                
                
                Movement=Movement1(iMovement,:);
                Mass=Mass1(iMass,:);
                Coils=Coils1(iCoils,:);
                Name=[Movement '_' Resistances(iResistace,:)]
                Res=Res2(iResistace,:);
                Name1=Name;
                
                %ACCELEROMETER
                xsens = importdata('p04_k30_50-000.txt');
                % xsens = importdata([Name '-000.txt']);
                if isequal(class(xsens),'struct')==1
                    xsens1=xsens.data;
                    xsens1(:,2)=[];
                    PacketCounter=xsens1(:,1);
                    Acc_X=xsens1(:,2);
                    Acc_Y=xsens1(:,3);
                    Acc_Z=xsens1(:,4);
                    FreeAcc_X=xsens1(:,5);
                    FreeAcc_Y=xsens1(:,6);
                    FreeAcc_Z=xsens1(:,7);
                    Gyr_X=xsens1(:,8);
                    Gyr_Y=xsens1(:,9);
                    Gyr_Z=xsens1(:,10);
                    VelInc_X=xsens1(:,11);
                    VelInc_Y=xsens1(:,12);
                    VelInc_Z=xsens1(:,13);
                    OriInc_q0=xsens1(:,14);
                    OriInc_q1=xsens1(:,15);
                    OriInc_q2=xsens1(:,16);
                    OriInc_q3=xsens1(:,17);
                    Roll=xsens1(:,18);
                    Pitch=xsens1(:,19);
                    Yaw=xsens1(:,20);
                    
                    dt=1/fts;
                    
%                     OriInc_q0=cumsum(OriInc_q0)*dt;
%                     OriInc_q1=cumsum(OriInc_q1)*dt;
%                     OriInc_q2=cumsum(OriInc_q2)*dt;
%                     OriInc_q3=cumsum(OriInc_q3)*dt;
%                     Yaw=(180/pi)*atan(2*(OriInc_q1.*OriInc_q2+OriInc_q0.*OriInc_q3)./(2*OriInc_q0.*OriInc_q0+2*OriInc_q1.*OriInc_q1-1));
%                     Pitch=-(180/pi)*asin(2*(OriInc_q1.*OriInc_q3-OriInc_q0.*OriInc_q2));
%                     Roll=(180/pi)*atan(2*(OriInc_q2.*OriInc_q3+OriInc_q0.*OriInc_q1)./(2*OriInc_q0.*OriInc_q0+2*OriInc_q3.*OriInc_q3-1));
                    
                    % Roll=Roll-mean(Roll(1:nCorr));
                    % Pitch=Pitch-mean(Pitch(1:nCorr));
                    % Yaw=Yaw-mean(Yaw(1:nCorr));
                    
                    n=length(PacketCounter);
                    %external trigger (stop record) set to t=19s
                    
                    ts = 0 : dt : n*dt;
                    ts = ts(1:end-1); ts=ts.';
                    
                    
                    %COILS
                    % rec = load([Name '.mat']);
                    % tc = rec.(Name).X.Data; tc=tc.';
                    % V = rec.(Name).Y(1).Data; V=-V.';

                    rec = load('p04_k30_4c.mat');
                    tc = rec.('p04_k30_4c').X.Data; tc=tc.';
                    V = rec.('p04_k30_4c').Y(1).Data; V=-V.';

                    %sync with tf=20s (simulation duration)
                    % tc = tc + ts(end)-tc(end); %?????????????
                    tc = tc + tdiff; %?????????????
                    Power=(V.^2)/Res;
                    AvPower=(sum(Power)-(1/2)*(Power(1)+Power(end)))/length(Power) % Output average power
                    MaxPower=max(Power)
                    
                    dt1=tc(2)-tc(1);
                    n2=length(tc);
                    df2=1/(n2*dt1);             % df = 1/(nt*dt) ~ f1/Ncicles; ffinal = (nt-1)/(nt*dt) ~ 1/dt = SampleRate*ff
                    f1V=(0:n2-1)*df2;
                    YV=fft(V(1:end-1)); % FFT
                    
                    Y_1V=2*YV(1:end/2,:)/n2;  %(DFT [Voltage])
                    f_1V=f1V(1:size(Y_1V,1)).';
                    
                    
                    
                    %Rotation parameters
                    
                    Acci=[Acc_X.';Acc_Y.';Acc_Z.'];  %RiI.(d2TI_dt + gI)
                    g_acelI=mean(Acci(:,1:nCorr),2) % Estimation of gravity acceleration
                    
                    [t1,y] = ode45(@(t,y) [[1,sin(y(1))*tan(y(2)),cos(y(1))*tan(y(2))]*interp1(ts,[Gyr_X,Gyr_Y,Gyr_Z],t).';[0,cos(y(1)),-sin(y(1))]*interp1(ts,[Gyr_X,Gyr_Y,Gyr_Z],t).';[0,sin(y(1))/cos(y(2)),cos(y(1))/cos(y(2))]*interp1(ts,[Gyr_X,Gyr_Y,Gyr_Z],t).'], ts, [0,0,0]);
                    Roll=y(:,1)*180/pi;
                    Pitch=y(:,2)*180/pi;
                    Yaw=y(:,3)*180/pi;
                    
                    psi=Yaw*pi/180;
                    theta=Pitch*pi/180;
                    phi=Roll*pi/180;
                    
                    
                    BIJ=eye(3);          % Unitary rotation matrix between inertial E' basis (origin in center of the container) and inertial E basis (origin on the accelerometer)
                    RiI=zeros(3,3,n);Wij=RiI; R1iI=RiI; AccFreei=zeros(3,n); d2TI_dt2=AccFreei; beta=zeros(n,1); alpha=beta; gamma=beta;
                    w=OriInc_q0; x=OriInc_q1; y=OriInc_q2; z=OriInc_q3;
                    for i=1:n
                        RiI(:,:,i)=[1,0,0;0,cos(phi(i)),sin(phi(i));0,-sin(phi(i)),cos(phi(i))]*[cos(theta(i)),0,-sin(theta(i));0,1,0;sin(theta(i)),0,cos(theta(i))]*[cos(psi(i)),sin(psi(i)),0;-sin(psi(i)),cos(psi(i)),0;0,0,1];
%                         RiI(:,:,i)=[-1+2*(w(i)^2+x(i)^2),2*(x(i)*y(i)-z(i)*w(i)),2*(x(i)*z(i)+y(i)*w(i));2*(x(i)*y(i)+z(i)*w(i)),-1+2*(w(i)^2+y(i)^2),2*(y(i)*z(i)-x(i)*w(i));2*(x(i)*z(i)-y(i)*w(i)),2*(y(i)*z(i)+x(i)*w(i)),-1+2*(w(i)^2+z(i)^2)];
                            
                        Wij(:,:,i)=[0,-Gyr_Z(i),Gyr_Y(i);Gyr_Z(i),0,-Gyr_X(i);-Gyr_Y(i),Gyr_X(i),0];
                        R1iI(:,:,i)=BIJ*RiI(:,:,i)*BIJ.';   % Rotation matrix for the container
                        
                        
                        AccFreei(:,i)=Acci(:,i)-(RiI(:,:,i)*g_acelI);  %RiI.d2TI/dt2
                        d2TI_dt2(:,i)=(RiI(:,:,i).')*AccFreei(:,i);
                        % AccFreei(:,i)dTI_
                        
                        beta(i)=acos(R1iI(3,3,i));
                        alpha(i)=atan2(R1iI(3,1,i),-R1iI(3,2,i));
                        gamma(i)=atan2(R1iI(1,3,i),R1iI(2,3,i));
                        
                    end
                    
                   
                    [t1,y] = ode45(@(t,y) [-sin(y(3)),cos(y(3)),0;-cos(y(3)),-sin(y(3)),0;cos(y(2))*sin(y(3)),-cos(y(2))*cos(y(3)),-sin(y(2))]*interp1(ts,[Gyr_X,Gyr_Y,Gyr_Z],t).' + [cos(y(2)),0,0;0,0,0;0,0,cos(y(2))]*[y(1);y(2);y(3)], ts, [0,0,0]);
                    alpha_sinbeta=y(:,1);
                    beta=y(:,2);
                    gamma_sinbeta=y(:,3);
        
                    tsMid=(ts(2:end)+ts(1:end-1))/2;
                    
                    Aread2TI_dt2=zeros(3,n-1); dWij_dt1=zeros(3,3,n-1); dRiI_dt=dWij_dt1; RiIint=dWij_dt1; Wij2=dWij_dt1;
                    alpha(1)=0;beta(1)=0;gamma(1)=0;
                    
                    dW3=gradient(Gyr_Z)/dt;
                    dW2=gradient(Gyr_Y)/dt;
                    dW1=gradient(Gyr_X)/dt;
                    

      
                    for i=1:n-1
                        Aread2TI_dt2(:,i)=(1/2)*(d2TI_dt2(:,i+1)+d2TI_dt2(:,i))*(ts(i+1)-ts(i));
%                         dWij_dt1(:,:,i)=(Wij(:,:,i+1)-Wij(:,:,i))/(ts(i+1)-ts(i));
                        
%                         RiI(:,:,i+1)=((RiI(:,:,i).')*(eye(3)+Wij(:,:,i)*dt)).';

                        dWij_dt1(:,:,i)=[0,-dW3(i),dW2(i);dW3(i),0,-dW1(i);-dW2(i),dW1(i),0];
                        
                        dRiI_dt(:,:,i)=(RiI(:,:,i+1)-RiI(:,:,i))/(ts(i+1)-ts(i));
                        RiIint(:,:,i)=(RiI(:,:,i+1)+RiI(:,:,i))/2;
                        Wij2(:,:,i)=RiIint(:,:,i)*(dRiI_dt(:,:,i).');
                        
                        if abs(sin(beta(i+1)))<=1e-1
                            alpha(i+1)=0;
                            gamma(i+1)=0;
                        else
                            alpha(i+1)=alpha_sinbeta(i+1)/sin(beta(i+1));
                            gamma(i+1)=gamma_sinbeta(i+1)/sin(beta(i+1));
                        end
                        
%                         if abs(R1iI(3,3,i+1))>0.999
%                             gamma(i+1)=gamma(i);
%                             alpha(i+1)=-gamma(i+1)+asin(R1iI(1,2,i+1));
%                             beta(i+1)=beta(i);
%                         else
%                             beta1(i+1)=acos(R1iI(3,3,i+1));
%                             beta2(i+1)=-acos(R1iI(3,3,i+1))+2*pi;
%                             alpha1(i+1)=asin(R1iI(3,1,i+1)/sin(beta1(i+1)));
%                             alpha2(i+1)=asin(R1iI(3,1,i+1)/sin(beta2(i+1)));
%                             gamma1(i+1)=atan2(R1iI(1,3,i+1)/sin(beta1(i+1)),R1iI(2,3,i+1)/sin(beta1(i+1)));
%                             gamma2(i+1)=atan2(R1iI(1,3,i+1)/sin(beta2(i+1)),R1iI(2,3,i+1)/sin(beta2(i+1)));
%                             if abs(gamma1(i+1)-gamma(i))<abs(gamma2(i+1)-gamma(i))
%                                 %     if R1iI(3,2,i+1)<=0;
%                                 beta(i+1)=beta1(i+1);
%                                 alpha(i+1)=alpha1(i+1);
%                                 gamma(i+1)=gamma1(i+1);
%                             else
%                                 beta(i+1)=beta2(i+1);
%                                 alpha(i+1)=alpha2(i+1);
%                                 gamma(i+1)=gamma2(i+1);
%                             end
%                         end
                        
                    end
                    
                    dWij_dt1(:,:,i+1)=[0,-dW3(i+1),dW2(i+1);dW3(i+1),0,-dW1(i+1);-dW2(i+1),dW1(i+1),0];
                    dWij_dt=dWij_dt1;
                    
                    T1I=zeros(3,n); RiI_d2RjI_dt2=zeros(3,3,n); movTFz_Fg_m=T1I; movtFz=T1I;
                    for i=1:n
%                         T1I(:,i)=BIJ*(TICorr(:,i)+(RiI(:,:,i).'-eye(3))*L_E1_E);
                        RiI_d2RjI_dt2(:,:,i)=Wij(:,:,i)*Wij(:,:,i)+Neuler*dWij_dt(:,:,i);
              
                        movTFz_Fg_m(:,i)=-BIJ*(Acci(:,i)+RiI_d2RjI_dt2(:,:,i)*L_E1_E);
                        movtFz(:,i)=-Fcent*BIJ*RiI_d2RjI_dt2(:,:,i)*BIJ(3,:).';
                        
%                              AccFreei(:,i)=Acci(:,i)-(RiI(:,:,i)*g_acelI);  %RiI.d2TI/dt2
                        d2TI_dt2(:,i)=-(RiI(:,:,i).')*movTFz_Fg_m(:,i)-g_acelI;
                        
                    end
                    
                    
%                     plot(ts, d2TI_dt2)
                    d2TI_dt2=d2TI_dt2.';
                    Y=fft(d2TI_dt2(1:end,:)); % FFT
                    %                     f_1(:)=f1(1:end/2);   % Ouput frequency
                    P2 = Y/n;
                    P1 = P2(1:floor(n/2+1),:);
                    P1(2:end-1,:) = 2*P1(2:end-1,:);
                    Y1=P1;               % Voltage FFT
                    Y1(1:nfilter,:)=0;
                    
                    P12=Y1;
                    P12(2:end-1,:) = (1/2)*P12(2:end-1,:);
                    P22=[P12;conj(flip(P12(2:ceil(n/2),:)))];
                    Y12=n*P22;
                    I12=ifft(Y12);                   % Filtered Voltage
                    d2TI_dt2=I12.';
%                     hold on
%                     plot(ts, d2TI_dt2)

                    for i=1:n-1
                        Aread2TI_dt2(:,i)=(1/2)*(d2TI_dt2(:,i+1)+d2TI_dt2(:,i))*(ts(i+1)-ts(i));
                    end
                    
                    alpha=alpha*(180/pi);
                    beta=beta*(180/pi);
                    gamma=gamma*(180/pi);
                    
                    dTI_dt=[zeros(3,1),cumsum(Aread2TI_dt2,2)];
                    % RiIint=interp1(ts,RiI,tsMid,'linear')
                    % dRiI=diff(RiI,1,3)
                    
%                     dWij_dt=permute(interp1(tsMid,permute(dWij_dt1,[3 1 2]),ts,'linear'),[2 3 1]);
%                     dWij_dt(:,:,1)=dWij_dt1(:,:,1); dWij_dt(:,:,end)=dWij_dt1(:,:,end);
                    
                    
%                     dWij_dt=permute(interp1(ts,permute(dWij_dt1,[3 1 2]),ts,'spline'),[2 3 1]);
%                     dWij_dt=smoothdata(dWij_dt);
                    
                    dTI_dtCorr=zeros(3,n);
                    
                    % dt=ts(2)-ts(1);
                    % f = (0:n/2)/(dt*n);
                    % Y=fft(dTI_dt.');
                    % P2 = Y/n;
                    % P1 = P2(1:n/2+1,:);
                    % P1(2:end-1,:) = 2*P1(2:end-1,:);
                    % Y1=P1;
                    %
                    % Cond=f>0.2;
                    % P12=Y1.*Cond.';
                    % P12(2:end-1,:) = (1/2)*P12(2:end-1,:);
                    % P22=[P12;conj(flip(P12(2:ceil(n/2),:),1))];
                    % Y12=n*P22;
                    % I12=ifft(Y12).';                   % Filtered Voltage
                    % dTI_dtCorr=I12;
                    
%                     npoly=10;
%                     dTI_dtCorr(1,:)=dTI_dt(1,:)-polyval(polyfit(ts,dTI_dt(1,:),npoly),ts).';
%                     dTI_dtCorr(2,:)=dTI_dt(2,:)-polyval(polyfit(ts,dTI_dt(2,:),npoly),ts).';
%                     dTI_dtCorr(3,:)=dTI_dt(3,:)-polyval(polyfit(ts,dTI_dt(3,:),npoly),ts).';
                    dTI_dtCorr(1,:)=dTI_dt(1,:);
                    dTI_dtCorr(2,:)=dTI_dt(2,:);
                    dTI_dtCorr(3,:)=dTI_dt(3,:);
                    %                     plot(ts, d2TI_dt2)
                    dTI_dtCorr=dTI_dtCorr.';
                    Y=fft(dTI_dtCorr(1:end,:)); % FFT
                    %                     f_1(:)=f1(1:end/2);   % Ouput frequency
                    P2 = Y/n;
                    P1 = P2(1:floor(n/2+1),:);
                    P1(2:end-1,:) = 2*P1(2:end-1,:);
                    Y1=P1;               % Voltage FFT
                    Y1(1:nfilter2,:)=0;
                    
                    P12=Y1;
                    P12(2:end-1,:) = (1/2)*P12(2:end-1,:);
                    P22=[P12;conj(flip(P12(2:ceil(n/2),:)))];
                    Y12=n*P22;
                    I12=ifft(Y12);                   % Filtered Voltage
                    dTI_dtCorr=I12.';
%                     hold on
%                     plot(ts, d2TI_dt2)
                    
                    
                    
                    AreadTI_dt=zeros(3,n-1);AreadTI_dtCorr=AreadTI_dt;
                    for i=1:n-1
                        AreadTI_dt(:,i)=(1/2)*(dTI_dt(:,i+1)+dTI_dt(:,i))*(ts(i+1)-ts(i));
                        AreadTI_dtCorr(:,i)=(1/2)*(dTI_dtCorr(:,i+1)+dTI_dtCorr(:,i))*(ts(i+1)-ts(i));
                        
                    end
                    TI=[zeros(3,1),cumsum(AreadTI_dt,2)];
                    TICorr=[zeros(3,1),cumsum(AreadTI_dtCorr,2)];
                    TI=TICorr;
                    T1I=TICorr;
                    
                    
                    
                    h=figure(1);
                    h.Name='Acc';
                    hold on
                    h=plot(ts,Acc_X,'r');
                    h.DisplayName=['x ' Name1];
                    h=plot(ts,Acc_Y,'g');
                    h.DisplayName=['y ' Name1];
                    h=plot(ts,Acc_Z,'b');
                    h.DisplayName=['z ' Name1];
                    legend
                    plot(ts,-movTFz_Fg_m(1,:),':m',ts,-movTFz_Fg_m(2,:),':k',ts,-movTFz_Fg_m(3,:),':c')
                    xlabel('t (s)')
                    ylabel('Acc (m/s^2)')
                    % plot(ts,AccI(1,:),'m',ts,AccI(2,:),'k',ts,AccI(3,:),'c')
                    
                    h=figure(2);
                    h.Name='FreeAcc';
                    hold on
                    h=plot(ts,FreeAcc_X,'r');
                    h.DisplayName=['x ' Name1];
                    h=plot(ts,FreeAcc_Y,'g');
                    h.DisplayName=['y ' Name1];
                    h=plot(ts,FreeAcc_Z,'b');
                    h.DisplayName=['z ' Name1];
                    legend
                    plot(ts,AccFreei(1,:),':m',ts,AccFreei(2,:),':k',ts,AccFreei(3,:),':c')
                    xlabel('t (s)')
                    ylabel('Acc (m/s^2)')
                    
                    h=figure(3);
                    h.Name='Orient';
                    hold on
                    h=plot(ts,Roll,'r');
                    h.DisplayName=['Roll ' Name1];
                    h=plot(ts,Pitch,'g');
                    h.DisplayName=['Pitch ' Name1];
                    h=plot(ts,Yaw,'b');
                    h.DisplayName=['Yaw ' Name1];
                    legend
                    xlabel('t (s)')
                    ylabel('Angle (º)')
                    
                    h=figure(4);
                    h.Name='W';
                    hold on
                    h=plot(ts,Gyr_X,'r');
                    h.DisplayName=['W_x ' Name1];
                    h=plot(ts,Gyr_Y,'g');
                    h.DisplayName=['W_y ' Name1];
                    h=plot(ts,Gyr_Z,'b');
                    h.DisplayName=['W_z ' Name1];
                    plot(tsMid,permute(-Wij2(2,3,:),[3 1 2]),':m',tsMid,permute(Wij2(1,3,:),[3 1 2]),':k',tsMid,permute(-Wij2(1,2,:),[3 1 2]),':c')
                    legend
                    xlabel('t (s)')
                    ylabel('W = Gyr (rad/s)')
                    
                    h=figure(5);
                    h.Name='dW/dt';
                    hold on
                    h=plot(ts,permute(-dWij_dt(2,3,:),[3 1 2]),'r');
                    h.DisplayName=['dW_x/dt ' Name1];
                    h=plot(ts,permute(dWij_dt(1,3,:),[3 1 2]),'g');
                    h.DisplayName=['dW_y/dt ' Name1];
                    h=plot(ts,permute(-dWij_dt(1,2,:),[3 1 2]),'b');
                    h.DisplayName=['dW_z/dt ' Name1];
                    legend
                    xlabel('t (s)')
                    ylabel('dW/dt (rad/s^2)')
                    
                    h=figure(6);
                    h.Name='d2TI/dt2';
                    hold on
                    h=plot(ts,d2TI_dt2(1,:),'r');
                    h.DisplayName=['d2T_X/dt2 ' Name1];
                    h=plot(ts,d2TI_dt2(2,:),'g');
                    h.DisplayName=['d2T_Y/dt2 ' Name1];
                    h=plot(ts,d2TI_dt2(3,:),'b');
                    h.DisplayName=['d2T_Z/dt2 ' Name1];
                    legend
                    xlabel('t (s)')
                    ylabel('d2TI/dt2 (m/s^2)')
                    
                    h=figure(7);
                    h.Name='dTI/dt';
                    hold on
                    h=plot(ts,dTI_dt(1,:),'r');
                    h.DisplayName=['dT_X/dt ' Name1];
                    h=plot(ts,dTI_dt(2,:),'g');
                    h.DisplayName=['dT_Y/dt ' Name1];
                    h=plot(ts,dTI_dt(3,:),'b');
                    h.DisplayName=['dT_Z/dt ' Name1];
                    plot(ts,dTI_dtCorr(1,:),'m:',ts,dTI_dtCorr(2,:),'k:',ts,dTI_dtCorr(3,:),'c:')
                    legend
                    xlabel('t (s)')
                    ylabel('dTI/dt (m/s)')
                    
                    h=figure(8);
                    h.Name='TI';
                    hold on
                    h=plot(ts,TI(1,:),'r');
                    h.DisplayName=['T_X ' Name];
                    h=plot(ts,TI(2,:),'g');
                    h.DisplayName=['T_Y ' Name];
                    h=plot(ts,TI(3,:),'b');
                    h.DisplayName=['T_Z ' Name1];
                    plot(ts,T1I(1,:),'m:',ts,T1I(2,:),'k:',ts,T1I(3,:),'c:')
                    legend
                    xlabel('t (s)')
                    ylabel('TI (m)')
                    
                    h=figure(9);
                    h.Name='Mov';
                    hold on
                    inde=1:size(T1I,2);
                    x=T1I(1,inde).';y=T1I(2,inde).';z=T1I(3,inde).';
                    xseg = [x(1:end-1),x(2:end)];
                    yseg = [y(1:end-1),y(2:end)];
                    zseg = [z(1:end-1),z(2:end)];
                    % h=plot3(T1I(1,:),T1I(2,:),T1I(3,:),'k');
                    h=plot3(xseg.',yseg.',zseg.','k');
                    segColors = jet(length(inde)-1); % Choose a colormap
                    set(h, {'Color'}, mat2cell(segColors,ones(length(inde)-1,1),3))
                    grid on
                    xlabel('X (m)')
                    ylabel('Y (m)')
                    zlabel('Z (m)')
                    % leng=1e-1;
                    % for i=1:50:n
                    % plot3([T1I(1,i),T1I(1,i)+leng*R1iI(1,1,i)],[T1I(2,i),T1I(2,i)+leng*R1iI(1,2,i)],[T1I(3,i),T1I(3,i)+leng*R1iI(1,3,i)],'r')
                    % plot3([T1I(1,i),T1I(1,i)+leng*R1iI(2,1,i)],[T1I(2,i),T1I(2,i)+leng*R1iI(2,2,i)],[T1I(3,i),T1I(3,i)+leng*R1iI(2,3,i)],'g')
                    % plot3([T1I(1,i),T1I(1,i)+leng*R1iI(3,1,i)],[T1I(2,i),T1I(2,i)+leng*R1iI(3,2,i)],[T1I(3,i),T1I(3,i)+leng*R1iI(3,3,i)],'b')
                    % end
                    nn=50;
                    quiver3(T1I(1,inde(1):nn:end),T1I(2,inde(1):nn:end),T1I(3,inde(1):nn:end),permute(R1iI(1,1,inde(1):nn:end),[1 3 2]),permute(R1iI(1,2,inde(1):nn:end),[1 3 2]),permute(R1iI(1,3,inde(1):nn:end),[1 3 2]),'r')
                    quiver3(T1I(1,inde(1):nn:end),T1I(2,inde(1):nn:end),T1I(3,inde(1):nn:end),permute(R1iI(2,1,inde(1):nn:end),[1 3 2]),permute(R1iI(2,2,inde(1):nn:end),[1 3 2]),permute(R1iI(2,3,inde(1):nn:end),[1 3 2]),'g')
                    quiver3(T1I(1,inde(1):nn:end),T1I(2,inde(1):nn:end),T1I(3,inde(1):nn:end),permute(R1iI(3,1,inde(1):nn:end),[1 3 2]),permute(R1iI(3,2,inde(1):nn:end),[1 3 2]),permute(R1iI(3,3,inde(1):nn:end),[1 3 2]),'b')
                    view([40 12])
                    axis equal
                    
                    h=figure(10);
                    h.Name='AccInert';
                    hold on
                    h=plot(ts,movTFz_Fg_m(1,:),'r');
                    h.DisplayName=['Acc_x ' Name1];
                    h=plot(ts,movTFz_Fg_m(2,:),'g');
                    h.DisplayName=['Acc_y ' Name1];
                    h=plot(ts,movTFz_Fg_m(3,:),'b');
                    h.DisplayName=['Acc_z ' Name1];
                    legend
                    xlabel('t (s)')
                    ylabel('Finert. accel. (m/s^2)')
                    
                    h=figure(11);
                    h.Name='AccCentriInert';
                    hold on
                    h=plot(ts,movtFz(1,:),'r');
                    h.DisplayName=['CentAcc_x ' Name1];
                    h=plot(ts,movtFz(2,:),'g');
                    h.DisplayName=['CentAcc_y ' Name1];
                    h=plot(ts,movtFz(3,:),'b');
                    h.DisplayName=['CentAcc_z ' Name1];
                    legend
                    xlabel('t (s)')
                    ylabel('Centrifugal accel. ((m/s^2)/m)')
                    
                    h=figure(12);
                    h.Name='Orient2';
                    hold on
                    h=plot(ts,alpha,'r');
                    h.DisplayName=['\alpha ' Name1];
                    h=plot(ts,beta,'g');
                    h.DisplayName=['\beta ' Name1];
                    h=plot(ts,gamma,'b');
                    h.DisplayName=['\gamma ' Name1];
                    legend
                    xlabel('t (s)')
                    ylabel('Angle (º)')
                    
                    h=figure(15);
                    h.Name='V';
                    hold on
                    h=plot(tc,-V);
                    h.DisplayName=['V ' Name1];
                    legend
                    xlabel('t (s)')
                    ylabel('V (V)')
                    
                    h=figure(16);
                    h.Name='I';
                    hold on
                    h=plot(tc,-V/Res);
                    h.DisplayName=['I ' Name1];
                    legend
                    xlabel('t (s)')
                    ylabel('I (A)')
                    
                    h=figure(17);
                    h.Name='P';
                    hold on
                    h=plot(tc,Power);
                    h.DisplayName=['P ' Name1];
                    legend
                    xlabel('t (s)')
                    ylabel('P (W)')
                    
                    h=figure(19);
                    h.Name='FFT(V)';
                    hold on
                    h=plot(f_1V,abs(Y_1V));
                    h.DisplayName=['V ' Name1];
                    legend
                    xlabel('f (Hz)')
                    ylabel('DFT[V] (V)')
                    
                    %% aaDynamics.m
                    if NumericalCalc==1
                        
                        if isequal(Coils,'2c')
                            load('a0Out_2C.mat')
                            Rc=8.41e+03;                  % Effective resistance of the coils in Ohm
                        elseif isequal(Coils,'4c')
                            load('a0Out_4C.mat')
                            Rc=2*8.41e+03;                  % Effective resistance of the coils in Ohm
                        end
                        
                        if isequal(Mass,'LW')
                            mMassi=MassMagnet;
                            Xcm=0;
                        elseif isequal(Mass,'HV')
                            mMassi=MassMagnet+MassWeight;
                            Xcm=Xcm3;
                        end
                        
                        Dampf1=Dampf;                 % f/m Static friction force in N/kg  (~=Dampf for Teflon)
                        Dampc_m=Dampc/mMassi;
                        
                        AbsTolOde=1e-6;               % Absolute error tolerance in ODE (default = 1e-6)
                        RelTolOde=1e-3;               % Relative error tolerance in ODE (default = 1e-3)
                        
                        y0 = [0,0,0];            % Initial conditions [delta - position, ddelta/dt - velocity, I - current]
                        fun = @(x) (movTFz_Fg_m(3,1)+Fmag_m_int(x,a0Out,mMassi));
                        d0=fzero(fun,0);             % steady state displacement of the magnet (magnetic force + gravity force = 0)
                        y0(1)=d0;
                        
                        opts = odeset('RelTol',RelTolOde,'AbsTol',AbsTolOde);
                        t2=tc;%linspace(ts(1),ts(end),10000);
                        n1=length(t2);
                        
                        % Low frequency approximation (wL << R) (Interpolate a0Out.mat from Magnetics.m) %
                        [t1,y] = ode45(@(t,y) [y(2); interp1(ts,movTFz_Fg_m(3,:),t,'linear','extrap')+(y(1)+Xcm)*interp1(ts,movtFz(3,:),t,'linear','extrap')+Fmag_m_int(y,a0Out,mMassi)+Flrz_m_int(y,a0Out,mMassi,Res,Rc)+Ffric_mf(y(2),Dampf,Dampc_m,Dampf1,interp1(ts,movTFz_Fg_m(3,:),t,'linear','extrap')+(y(1)+Xcm)*interp1(ts,movtFz(3,:),t,'linear','extrap')+Fmag_m_int(y,a0Out,mMassi)+Flrz_m_int(y,a0Out,mMassi,Res,Rc))], t2, y0(1:2), opts);
                        y(:,3)=-y(:,2).*(interp1(a0Out(:,1),a0Out(:,3),y(:,1),'linear','extrap')./(Res+Rc)); %Current
                        y(:,4)=Res*y(:,3);  % Voltage
                        Pow=y(:,3).*y(:,4); % Output power vs time (I*V)
                        AvPow=(sum(Pow)-(1/2)*(Pow(1)+Pow(end)))/n1; % Output average power
                   
                        dt=t2(2)-t2(1);
                        df=1/(n1*dt);             % df = 1/(nt*dt) ~ f1/Ncicles; ffinal = (nt-1)/(nt*dt) ~ 1/dt = SampleRate*ff
                        f1=(0:n1-1)*df;
                        Y=fft(y(1:end-1,:)); % FFT
                        
                        Y_1=2*Y(1:end/2,:)/n1;  %(DFT [Displacement; Velocity; Current; Voltage])
                        f_1=f1(1:size(Y_1,1));
                        
                        
                        h=figure(13);
                        h.Name='d';
                        hold on
                        h=plot(t1,y(:,1),'--');
                        h.DisplayName=['d ' Name1];
                        legend
                        xlabel('t (s)')
                        ylabel('d (m)')
                        
                        h=figure(14);
                        h.Name='dd/dt';
                        hold on
                        h=plot(t1,y(:,2),'--');
                        h.DisplayName=['dd/dt ' Name1];
                        legend
                        xlabel('t (s)')
                        ylabel('dd/dt (m/s)')
                        
                        figure(16);
                        plot(t1,y(:,3),'--')
                        
                        figure(17);
                        plot(t1,Pow,'--')
                        
                        h=figure(18);
                        h.Name='FFT(d)';
                        hold on
                        h=plot(f_1,abs(Y_1(:,1)),'--');
                        h.DisplayName=['d ' Name1];
                        legend
                        xlabel('f (Hz)')
                        ylabel('DFT[d] (m)')
                        
                        figure(19);
                        plot(f_1,abs(Y_1(:,4)),'--');
                        
                        figure(15);
                        plot(t1,y(:,4),'--')
                    end
                    
                    if Animation==1
                        h=figure(9);
                        ax=[h.Children.XLim,h.Children.YLim,h.Children.ZLim]*1.6;
                        figure(20)
                        for ind=1:n
                            hold off
                            plot3(T1I(1,1:ind),T1I(2,1:ind),T1I(3,1:ind));
                            hold on
                            plot3(T1I(1,1:ind),T1I(2,1:ind),0*T1I(3,1:ind),'b--');
                            quiver3(T1I(1,ind),T1I(2,ind),T1I(3,ind),permute(R1iI(1,1,ind),[1 3 2]),permute(R1iI(1,2,ind),[1 3 2]),permute(R1iI(1,3,ind),[1 3 2]),'r','LineWidth',1,'AutoScaleFactor',0.15)
                            quiver3(T1I(1,ind),T1I(2,ind),T1I(3,ind),permute(R1iI(2,1,ind),[1 3 2]),permute(R1iI(2,2,ind),[1 3 2]),permute(R1iI(2,3,ind),[1 3 2]),'g','LineWidth',1,'AutoScaleFactor',0.15)
                            quiver3(T1I(1,ind),T1I(2,ind),T1I(3,ind),permute(R1iI(3,1,ind),[1 3 2]),permute(R1iI(3,2,ind),[1 3 2]),permute(R1iI(3,3,ind),[1 3 2]),'b','LineWidth',1,'AutoScaleFactor',0.15)
                            plot3(T1I(1,ind)+R1iI(3,1,ind)*[5,25]*1e-3,T1I(2,ind)+R1iI(3,2,ind)*[5,25]*1e-3,T1I(3,ind)+R1iI(3,3,ind)*[5,25]*1e-3,'m','LineWidth',2);
                            plot3(T1I(1,ind)+R1iI(3,1,ind)*[-5,-25]*1e-3,T1I(2,ind)+R1iI(3,2,ind)*[-5,-25]*1e-3,T1I(3,ind)+R1iI(3,3,ind)*[-5,-25]*1e-3,'m','LineWidth',2);
                            plot3(T1I(1,ind)+R1iI(3,1,ind)*[30,50]*1e-3,T1I(2,ind)+R1iI(3,2,ind)*[30,50]*1e-3,T1I(3,ind)+R1iI(3,3,ind)*[30,50]*1e-3,'m','LineWidth',2);
                            plot3(T1I(1,ind)+R1iI(3,1,ind)*[-30,-50]*1e-3,T1I(2,ind)+R1iI(3,2,ind)*[-30,-50]*1e-3,T1I(3,ind)+R1iI(3,3,ind)*[-30,-50]*1e-3,'m','LineWidth',2);
                            if NumericalCalc==1
                                d1=R1iI(3,:,ind)*interp1(t1,y(:,1),ts(ind),'linear','extrap');
                                d1dt=R1iI(3,:,ind)*interp1(t1,y(:,2),ts(ind),'linear','extrap');
                                plot3(T1I(1,ind)+d1(1),T1I(2,ind)+d1(2),T1I(3,ind)+d1(3),'.k','MarkerSize',20);
                                % quiver3(T1I(1,ind),T1I(2,ind),T1I(3,ind),d1dt(1),d1dt(2),d1dt(3),'y','LineWidth',1,'AutoScaleFactor',0.5);
                            end
                            grid on
                            xlabel('X (m)')
                            ylabel('Y (m)')
                            zlabel('Z (m)')
                            view([40 12])
                            axis equal
                            axis(ax)
                            drawnow
                            % pause(0.5)
                        end
                    end
                    
                else
                    'missing file'
                end
                
                x0 = [-1.5];
func=@(x)(sum(((-interp1(tc,V,tc+x,'linear','extrap')-y(:,4)).*(tc+x<=tc(end)).*(tc+x>=0)).^2));
% options = optimset('TolX',1e-10);
bestx = -fminsearch(@(x)func(x),x0)
            end
        end
    end
end