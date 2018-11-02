function OpticalFlowOutput(OutName,MovieName,BinaryMask,BoxSize,BlurSTD,ArrowSize,scale,dt)

    [ X,Y,Vx,Vy,mov ] = OpticalFlow( MovieName,BinaryMask,BoxSize,BlurSTD,ArrowSize,scale,dt );
    
    csvwrite(strcat(OutName,'_X.csv'),X);
    disp(size(X));
    csvwrite(strcat(OutName,'_Y.csv'),Y);
    disp(size(Y));
    csvwrite(strcat(OutName,'_Vx.csv'),Vx);
    disp(size(Vx));
    csvwrite(strcat(OutName,'_Vy.csv'),Vy);
    disp(size(Vy));

end