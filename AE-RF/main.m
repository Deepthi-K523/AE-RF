clc;clear
interaction = load(["C:\Users\deept\Desktop\cRNA-Functional\Ci-Di 583,88(functional sim_finder)\ci-di 585,88,650 new.txt"]);
disease_ss  = load(["C:\Users\deept\Desktop\cRNA-Functional\Ci-Di 583,88(functional sim_finder)\dis simlrty new.txt"]);

[nc,nd ]= size(interaction);

ciRNA_ss = miRNASS( interaction, disease_ss ); % Finding ciRNA functional similarity



[CC,DD]=gkl(nc,nd,interaction);

circ_feature = zeros(nc,nc);



for i = 1:nc
    for j = 1:nc
        if ciRNA_ss(i,j)~=0
            circ_feature(i,j) = ciRNA_ss(i,j); % Functional
        else 
            circ_feature(i,j) = CC(i,j); %Gaussian
        end
    end
end

dis_feature = zeros(nd,nd);

for i = 1:nd
    for j = 1:nd
        if disease_ss(i,j)~=0
            dis_feature(i,j) = disease_ss(i,j); % Functional
        else 
            dis_feature(i,j) = DD(i,j);  % Gaussian
        end
    end
end

%%Comma format


% dlmwrite('C:\Users\deept\Desktop\cRNA functional similarity-585,585.txt',circ_feature);
% dlmwrite('C:\Users\deept\Desktop\dis similarity-88,88.txt',dis_feature);

%%Tab format 
dlmwrite('C:\Users\deept\Desktop\integrated circRNA similarity.txt',circ_feature,'delimiter','\t');
dlmwrite('C:\Users\deept\Desktop\integrated disease similarity.txt',dis_feature,'delimiter','\t');