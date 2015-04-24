/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experimentshell;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Debug.Random;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;


/**
 *
 * @author Harvey
 */
public class ExperimentShell {

    /**
     * @param args the command line arguments
     * @throws java.lang.Exception
     */
    public static void main(String[] args) throws Exception {
                
        DataSource source;
        source = new DataSource("C:\\Users\\Harvey\\Documents\\iris.csv");
        Instances data = source.getDataSet();
        
        if (data.classIndex() == -1){
            data.setClassIndex(data.numAttributes() - 1);
        }
        
        data.randomize(new Random(1));
        
        RemovePercentage trainFilter = new RemovePercentage();
        trainFilter.setPercentage(70);
        trainFilter.setInputFormat(data);
        Instances train = Filter.useFilter(data, trainFilter);
        
        trainFilter.setInvertSelection(true);
        trainFilter.setInputFormat(data);
        Instances test = Filter.useFilter(data, trainFilter);
        
        Classifier hcClassifier = new HardCodedClassifier();
        hcClassifier.buildClassifier(train);
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(hcClassifier, test);
        System.out.println(eval.toSummaryString("\nResults\n-------------\n", false));
        
    }
    
}
