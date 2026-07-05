clear;
clc;
close all;

folders = {'224','227','256','331'};

Results = table();

for f = 1:length(folders)

    folderName = folders{f};
    zipName = [folderName '.zip'];

    fprintf('\n==============================\n');
    fprintf('Training GoogLeNet on folder: %s\n', folderName);
    fprintf('==============================\n');

    if isfile(zipName) && ~isfolder(folderName)
        unzip(zipName);
    end

    imds = imageDatastore(folderName, ...
        'IncludeSubfolders',true, ...
        'LabelSource','foldernames');

    [imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

    net = googlenet;

    inputSize = net.Layers(1).InputSize;

    lgraph = layerGraph(net);

    [learnableLayer,classLayer] = findLayersToReplace(lgraph);

    numClasses = numel(categories(imdsTrain.Labels));

    if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
        newLearnableLayer = fullyConnectedLayer(numClasses, ...
            'Name','new_fc', ...
            'WeightLearnRateFactor',10, ...
            'BiasLearnRateFactor',10);

    elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
        newLearnableLayer = convolution2dLayer(1,numClasses, ...
            'Name','new_conv', ...
            'WeightLearnRateFactor',10, ...
            'BiasLearnRateFactor',10);
    end

    lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

    newClassLayer = classificationLayer('Name','new_classoutput');
    lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

    pixelRange = [-30 30];
    scaleRange = [0.9 1.1];

    imageAugmenter = imageDataAugmenter( ...
        'RandXReflection',true, ...
        'RandXTranslation',pixelRange, ...
        'RandYTranslation',pixelRange, ...
        'RandXScale',scaleRange, ...
        'RandYScale',scaleRange);

    augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
        'DataAugmentation',imageAugmenter);

    augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

    miniBatchSize = 13;
    valFrequency = floor(numel(imdsTrain.Files)/miniBatchSize);

    options = trainingOptions('sgdm', ...
        'MiniBatchSize',miniBatchSize, ...
        'MaxEpochs',15, ...
        'InitialLearnRate',3e-4, ...
        'Shuffle','every-epoch', ...
        'ValidationData',augimdsValidation, ...
        'ValidationFrequency',valFrequency, ...
        'Verbose',false, ...
        'Plots','training-progress');

    trainingStart = tic;
    net = trainNetwork(augimdsTrain,lgraph,options);
    trainingTime = toc(trainingStart);

    [YPred,probs] = classify(net,augimdsValidation);

    trueLabels = imdsValidation.Labels;
    accuracy = mean(YPred == trueLabels);

    figure
    idx = randperm(numel(imdsValidation.Files),min(4,numel(imdsValidation.Files)));
    for i = 1:length(idx)
        subplot(2,2,i)
        I = readimage(imdsValidation,idx(i));
        imshow(I)
        label = YPred(idx(i));
        title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
    end
    saveas(gcf,['Predictions_GoogLeNet_' folderName '.png']);

    confMat = confusionmat(trueLabels,YPred);

    figure
    confusionchart(trueLabels,YPred);
    title(['GoogLeNet Confusion Matrix - ' folderName]);
    saveas(gcf,['ConfusionMatrix_GoogLeNet_' folderName '.png']);

    tp = diag(confMat);
    fp = sum(confMat,1)' - tp;
    fn = sum(confMat,2) - tp;
    tn = sum(confMat(:)) - (tp + fp + fn);

    precision = tp ./ (tp + fp);
    recall = tp ./ (tp + fn);
    specificity = tn ./ (tn + fp);
    f1Score = 2 * (precision .* recall) ./ (precision + recall);

    precision(isnan(precision)) = 0;
    recall(isnan(recall)) = 0;
    specificity(isnan(specificity)) = 0;
    f1Score(isnan(f1Score)) = 0;

    over_all_precision = mean(precision);
    over_all_recall = mean(recall);
    over_all_specificity = mean(specificity);
    over_all_f1Score = mean(f1Score);

    save(['GoogLeNet_' folderName '.mat'],'net');

    newRow = table( ...
        string("GoogLeNet"), ...
        string(folderName), ...
        accuracy, ...
        over_all_precision, ...
        over_all_recall, ...
        over_all_specificity, ...
        over_all_f1Score, ...
        trainingTime, ...
        'VariableNames', {'Network','Folder','Accuracy','Precision','Recall','Specificity','F1Score','TrainingTime_seconds'});

    Results = [Results; newRow];

    fprintf('Finished folder %s\n', folderName);
    fprintf('Accuracy: %.4f\n', accuracy);
    fprintf('F1 Score: %.4f\n', over_all_f1Score);

end

disp(Results)

writetable(Results,'GoogLeNet_All_Folders_Results.xlsx');

fprintf('\nAll folders finished.\n');
fprintf('Results saved to GoogLeNet_All_Folders_Results.xlsx\n');
