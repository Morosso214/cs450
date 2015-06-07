/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnetwork;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Debug;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;
import weka.filters.unsupervised.instance.RemovePercentage;

/**
 *
 * @author Harvey
 */
public class NeuralNetwork {
    
    /**
     * @param args the command line arguments
     * @throws java.lang.Exception
     */
    public static void main(String[] args) throws Exception {
       
        ConverterUtils.DataSource source;
        source = new ConverterUtils.DataSource("C:\\Users\\Harvey\\Documents\\iris.csv");
        Instances data = source.getDataSet();
        
        if (data.classIndex() == -1){
            data.setClassIndex(data.numAttributes() - 1);
        }       
        
        data.randomize(new Debug.Random(1));
        
        RemovePercentage trainFilter = new RemovePercentage();
        trainFilter.setPercentage(70);
        trainFilter.setInputFormat(data);
        Instances train = Filter.useFilter(data, trainFilter);
        
        trainFilter.setInvertSelection(true);
        trainFilter.setInputFormat(data);
        Instances test = Filter.useFilter(data, trainFilter);
        
        Standardize filter = new Standardize();
        filter.setInputFormat(train);
        
        Instances newTrain = Filter.useFilter(test, filter);
        Instances newTest = Filter.useFilter(train, filter);
                
        Classifier nNet = new NeuralNet();
        nNet.buildClassifier(newTrain);
        Evaluation eval = new Evaluation(newTest);
        eval.evaluateModel(nNet, newTest);
        System.out.println(eval.toSummaryString("\nResults\n-------------\n", false));
    }
    
}