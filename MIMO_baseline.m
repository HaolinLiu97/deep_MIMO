RX_antenna=2;
TX_antenna=2;
n=10000;
interval=100;

BER_list=[];
SNRsdB=0:1:40;                             %Signal to Noise Ratio (in dB)
SNRs=10.^(SNRsdB/10);
SNR=SNRs;

for i=1:length(SNRsdB)
    No=1/SNR(i)*2;
    dataIn= randi([0 1],2,n*interval);
    dataOut=zeros(2,n*interval);
    for j=1:n
        h=sqrt(1/2)*randn([RX_antenna TX_antenna])+complex(0,sqrt(1/2)*randn([RX_antenna TX_antenna]));
        AWGN=sqrt(1/2).*(sqrt(No)*randn(2,interval)+complex(0,sqrt(No)*randn(2,interval)));
        %noise_power=mean(real(AWGN).^2,'all')+mean(imag(AWGN).^2,"all");
        dataMod = (dataIn(:,interval*(j-1)+1:interval*j)-0.5)*2; %simulate BPSK signal
    
        W=inv(h'*h+No*eye(2))*h';
    
        y_hat=h*(dataMod)+AWGN;
        y=real(W*y_hat);
        for k=1:interval
            if y(1,k)>0
                dataOut(1,interval*(j-1)+k)=1;
            else
                dataOut(1,interval*(j-1)+k)=0;
            end
            if y(2,k)>0
                dataOut(2,interval*(j-1)+k)=1;
            else
                dataOut(2,interval*(j-1)+k)=0;
            end
        end
    end
    error_count=sum(abs(dataOut-dataIn),'all');
    BER=error_count/n/interval;
    MMSE_BER_list(i)=BER;
end

for i=1:length(SNRsdB)
    No=1/SNR(i)*2;
    dataIn= randi([0 1],2,n*interval);
    dataOut=zeros(2,n*interval);
    for j=1:n
        h=sqrt(1/2)*randn([RX_antenna TX_antenna])+complex(0,sqrt(1/2)*randn([RX_antenna TX_antenna]));
        AWGN=sqrt(1/2).*(sqrt(No)*randn(2,interval)+complex(0,sqrt(No)*randn(2,interval)));
        %noise_power=mean(real(AWGN).^2,'all')+mean(imag(AWGN).^2,"all");
        dataMod = (dataIn(:,interval*(j-1)+1:interval*j)-0.5)*2; %simulate BPSK signal
    
        [U,S,V]=svd(h);
        x_hat=V*dataMod;
    
        y_hat=h*x_hat+AWGN;
        y=real(inv(S)*U'*y_hat);
        for k=1:interval
            if y(1,k)>0
                dataOut(1,interval*(j-1)+k)=1;
            else
                dataOut(1,interval*(j-1)+k)=0;
            end
            if y(2,k)>0
                dataOut(2,interval*(j-1)+k)=1;
            else
                dataOut(2,interval*(j-1)+k)=0;
            end
        end
    end
    error_count=sum(abs(dataOut-dataIn),'all');
    BER=error_count/n/interval;
    SVD_BER_list(i)=BER;
end

content=load('BER_deep_learning_MIMO.mat')
neural_MIMO_BER=content.BER;
neural_MIMO_BER
semilogy(SNRsdB,MMSE_BER_list,'b*');
hold on
semilogy(SNRsdB,neural_MIMO_BER,'r*');
semilogy(SNRsdB,SVD_BER_list,'g*');
%semilogy(SNRsdB_th,SER_th,'b');
title('BER versus SNR for 2x2 MIMO system');
xlabel('SNR');
ylabel('Bit error probability');
legend('MMSE Baseline','deep learning based MIMO','SVD Baseline');
xlim([0 40]);
grid on;