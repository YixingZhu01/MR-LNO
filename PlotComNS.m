NG=64;
x=-1:1/NG:1-1/NG;
y=-1:1/NG:1-1/NG;
[X,Y]=meshgrid(x,y);
WriteData=false;
Plotu=true;
interval=5;
in=1;
% i_list=[floor(20/interval),floor(50/interval),floor(100/interval),floor(200/interval),floor(500/interval)];
i_list=1:1:floor(500/interval);
TestNum=25;
error_origin=zeros(length(i_list),TestNum);
error_RMSE=zeros(length(i_list),TestNum);
error_rho=zeros(length(i_list),TestNum);
error_T=zeros(length(i_list),TestNum);
error_relative=zeros(length(i_list),TestNum);
error=zeros(length(i_list),1);
rho_max=zeros(TestNum,1);
rho_min=zeros(TestNum,1);
rho_avg=zeros(TestNum,1);
rho_sigma=zeros(TestNum,1);
T_max=zeros(TestNum,1);
T_min=zeros(TestNum,1);
T_avg=zeros(TestNum,1);
T_sigma=zeros(TestNum,1);
for round=1:TestNum
    load(['Data\ComNS128Re100Ma2_',num2str(round),'.mat']);
    
    if(WriteData==true)
        label_u=reshape(u(1,:),[NG*2,NG*2])';
        label_v=reshape(v(1,:),[NG*2,NG*2])';
        label_T=reshape(T(1,:),[NG*2,NG*2])';
        label_rho=reshape(rho(1,:),[NG*2,NG*2])';
        fid2=fopen(['InitialField_case',num2str(round),'.dat'],'w+');
        fprintf(fid2,'TITLE=ComNS\r\n');
        fprintf(fid2,'VARIABLES=\"X\",\"Y\",\"rho_label\",\"T_label\",\"u_label\",\"v_label\"\r\n');
        fprintf(fid2,'ZONE I=%d, J=%d, F=POINT\r\n',NG*2,NG*2);
        for k=1:NG*2*NG*2
            fprintf(fid2,'%f %f %f %f %f %f\r\n',X(k),Y(k),label_rho(k),label_T(k),label_u(k),label_v(k));
        end
        fclose(fid2);
    end
    rho_max(round)=max(max(log(rho(1,:))));
    rho_min(round)=min(min(log(rho(1,:))));
    rho_avg(round)=sum(sum(log(rho(1,:))))/length(rho(1,:));
    rho_sigma(round)=sum(sum((log(rho(1,:))+0.0015).*(log(rho(1,:))+0.0015)))/length(rho(1,:));
    T_max(round)=max(max(log(T(1,:))));
    T_min(round)=min(min(log(T(1,:))));
    T_avg(round)=sum(sum(log(T(1,:))))/length(rho(1,:));
    T_sigma(round)=sum(sum((log(T(1,:))-0.0268).*(log(T(1,:))-0.0268)))/length(rho(1,:));
    u_max(round)=max(max(u(1,:)));
    u_min(round)=min(min(u(1,:)));
    u_avg(round)=sum(sum(u(1,:)))/length(u(1,:));
    u_sigma(round)=sum(sum((u(1,:)+0).*(u(1,:)+0)))/length(u(1,:));
    v_max(round)=max(max(v(1,:)));
    v_min(round)=min(min(v(1,:)));
    v_avg(round)=sum(sum(v(1,:)))/length(v(1,:));
    v_sigma(round)=sum(sum((v(1,:)+0).*(v(1,:)+0)))/length(v(1,:));
    
    for kk=1:length(i_list)
        i=i_list(kk);
        output_u=reshape(output(i,round,1,:,:),[2*NG,2*NG]);
        output_v=reshape(output(i,round,2,:,:),[2*NG,2*NG]);
        output_rho=exp(reshape(output(i,round,3,:,:),[2*NG,2*NG]));
        output_T=exp(reshape(output(i,round,4,:,:),[2*NG,2*NG]));
        label_u=reshape(u(1+(i+0)*interval,:),[2*NG,2*NG])';
        label_v=reshape(v(1+(i+0)*interval,:),[2*NG,2*NG])';
        label_rho=reshape(rho(1+(i+0)*interval,:),[2*NG,2*NG])';
        label_T=reshape(T(1+(i+0)*interval,:),[2*NG,2*NG])';
        
%         if(kk==1)
            up=max(max(max(label_u)),max(max(output_u)));
            down=min(min(min(label_u)),min(min(output_u)));
            k=10;
            bar=down:(up-down)/k:up;
%         end
%         if(Plotu==true)
%             figure(5)
%             [C1,h1]=contourf(X,Y,output_u,bar);
%             figure(4)
%             [C2,h2]=contourf(X,Y,label_u,bar);
%             drawnow
%         else
%             figure(1);
%             clf;
%             startx=rand(100,1)*2-1;
%             starty=rand(100,1)*2-1;
%             streamline(X,Y,label_u,label_v,startx,starty);
%         end

        
        error_origin(kk,round)=(sqrt(sum(sum((output_u-label_u).*(output_u-label_u)))/sum(sum(label_u.*label_u)))+sqrt(sum(sum((output_v-label_v).*(output_v-label_v)))/sum(sum(label_v.*label_v))))*0.5;
        error_RMSE(kk,round)=sqrt((sum(sum((output_u-label_u).*(output_u-label_u)))+sum(sum((output_v-label_v).*(output_v-label_v))))/2/NG/2/NG);%均方根误差RMSE
        error_relative(kk,round)=sqrt((sum(sum((output_u-label_u).*(output_u-label_u)))+sum(sum((output_v-label_v).*(output_v-label_v))))/(sum(sum((label_u).*(label_u)))+sum(sum((label_v).*(label_v)))));%均方根误差RMSE
        
        error_rho(kk,round)=mean(mean(abs(output_rho-label_rho)));
        error_T(kk,round)=mean(mean(abs(output_T-label_T)));
        
        error(kk)=error(kk)+sqrt((sum(sum((output_u-label_u).*(output_u-label_u)))+sum(sum((output_v-label_v).*(output_v-label_v))))/2/NG/2/NG)/10;
        if(WriteData==true)
            fid2=fopen(['ComNSRe100Ma2_case',num2str(round),'_FNO_',num2str(i),'.dat'],'w+');
            fprintf(fid2,'TITLE=ComNS\r\n');
            fprintf(fid2,'VARIABLES=\"X\",\"Y\",\"rho\",\"T\",\"u\",\"v\",\"rho_label\",\"T_label\",\"u_label\",\"v_label\"\r\n');
            fprintf(fid2,'ZONE I=%d, J=%d, F=POINT\r\n',NG*2,NG*2);
            for k=1:NG*2*NG*2
                fprintf(fid2,'%f %f %f %f %f %f %f %f %f %f\r\n',X(k),Y(k),output_rho(k),output_T(k),output_u(k),output_v(k),label_rho(k),label_T(k),label_u(k),label_v(k));
            end
            fclose(fid2);
        end
        
       lkk=0;
       pause(0.1)
    end
end
error_origin_ave=sum(error_origin')/TestNum;
error_origin_ave(:,1)=error_origin_ave(:,1)/interval/TestNum;

error_RMSE_ave=sum(error_RMSE')/TestNum;
error_RMSE_ave=[error_RMSE_ave;min(error_RMSE');max(error_RMSE')];
% error_RMSE_ave=error_RMSE_ave';

error_relative_ave=sum(error_relative')/10;
error_relative_ave=[error_relative_ave;min(error_relative');max(error_relative')];
% error_relative_ave=error_relative_ave';

error_rho_ave=sum(error_rho')/TestNum;
error_rho_ave=[error_rho_ave;min(error_rho');max(error_rho')];
error_T_ave=sum(error_T')/TestNum;
error_T_ave=[error_T_ave;min(error_T');max(error_T')];

error_list=[error_rho_ave(1,:);zeros(1,5);error_T_ave(1,:);zeros(1,5);error_RMSE_ave(1,:);zeros(1,5)];


