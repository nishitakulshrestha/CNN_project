outputFolder= fullfile('com_vision');
rootFolder = fullfile(outputFolder,'101_ObjectCategories');
categories = {'butterfly','faces','sunflower'};
imds= imageDatastore(fullfile(rootFolder,categories),'LabelSource','foldernames');
tb1 = countEachLabel(imds);%count no. of images
minSetCount= min(tb1{:,2});
imds= splitEachLabel(imds , minSetCount,'randomize');
tb2 = countEachLabel(imds);

butterfly= find(imds.Labels == 'butterfly', 1);
faces= find(imds.Labels == 'faces', 1);
sunflower= find(imds.Labels =='sunflower', 1);
 figure
 subplot(2,2,1);
 imshow(readimage(imds,butterfly));
 %subplot(2,2,2);
 %imshow(readimage(imds,faces));
subplot(2,2,3);
imshow(readimage(imds,sunflower));

net = resnet50();
figure
plot (net)
title('Aechitecture of ResNet-50');
set(gca, 'YLim', [150 170]);% see the structure clearly

net.Layers(1)
net.Layers(end)
numel(net.Layers(end).ClassNames)% no. of classes in this n/w

[trainingSet, testSet]= splitEachLabel(imds, 0.3, 'randomize');%30 for training
imageSize= net.Layers(1).InputSize;%to find image size
augmentedTrainingSet= augmentedImageDatastore (imageSize,trainingSet,'ColorPreprocessing','gray2rgb');

augmentedTestSet= augmentedImageDatastore(imageSize,testSet,'ColorPreprocessing','gray2rgb');% to resize and convert gray scale image to rgb

w1 = net.Layers(2).Weights;% find weight of 2 conv. layer
w1= mat2gray(w1);%convert matrix to image
figure
 montage(w1)
 title('first convolutional layer weight')

featureLayer= 'fc1000';%name of layer exactly before classification layer
trainingFeatures= activations(net,augmentedTrainingSet, featureLayer, 'MiniBatchSize', 32,'OutputAs', 'columns');%activation function use gpu if gpu is not
%available cpu is used Minibach size=32 to ensure image size fits in gpu memory if run out then lower minibatchsize 

trainingLabels= trainingSet.Labels;%to train svm
classifier= fitcecoc(trainingFeatures, trainingLabels,'Learner','Linear','Coding','onevsall','ObservationsIn','columns');
%return full trained multiclass error correcting output model 
testFeatures= activations(net,augmentedTestSet, featureLayer, 'MiniBatchSize', 32,'OutputAs', 'columns');
predictLabels = predict(classifier, testFeatures,'ObservationsIn', 'columns');
testLabels= testSet.Labels;%
confMat=confusionmat(testLabels,predictLabels);
confMat=bsxfun(@rdivide,confMat, sum(confMat,2));

mean(diag(confMat))

newImage= imread(fullfile('test201.jpg'));
ds= augmentedImageDatastore(imageSize,newImage,'ColorPreprocessing','gray2rgb');
imageFeatures= activations(net,ds, featureLayer, 'MiniBatchSize', 32,'OutputAs','columns');
label = predict(classifier, imageFeatures,'ObservationsIn','columns');
figure
 subplot(2,2,1);
 imshow(readimage(imds,butterfly));
title(sprintf('Loaded class belongs to %s class',label))


newImage= imread(fullfile('test203.jpg'));
ds= augmentedImageDatastore(imageSize,newImage,'ColorPreprocessing','gray2rgb');
imageFeatures= activations(net,ds, featureLayer, 'MiniBatchSize', 32,'OutputAs','columns');
label = predict(classifier, imageFeatures,'ObservationsIn','columns');
figure
subplot(2,2,1);
imshow(readimage(imds,sunflower));
title(sprintf('Loaded class belongs to %s class',label))

