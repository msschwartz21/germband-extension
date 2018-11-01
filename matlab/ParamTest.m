function output = ParamTest(MovieName,boxmin,boxmax,boxstep,blurmin,blurmax,blurn,ArrowSize)
% Tests the parameters BoxSize and BlurSTD
% Accepts a value for ArrowSize
% All other inputs for OpticalFlow.m are set to 1 or None

    % Optical flow params
    scale = 0.5;
    dt = 1;

    blurs = linspace(blurmin,blurmax,blurn);
    boxes = boxmin:boxstep:boxmax;
    
    disp(blurs);
    disp(boxes);
    
    for i = 1:length(blurs)
        BlurSTD = blurs(i);
        for j = 1:length(boxes)
            BoxSize = boxes(j);
            disp(BlurSTD);
            disp(BoxSize);
%             [X,Y,Vx,Vy,Mov] = OpticalFlow (MovieName,[],BoxSize,BlurSTD,ArrowSize,scale,dt,'none');
%             
%             obj = VideoWriter(strcat('box',num2str(BoxSize),'blur',num2str(BlurSTD)));
%             obj.open();
%             obj.writeVideo(Mov);
%             obj.close();
            disp('done');
            
        end
    end

end