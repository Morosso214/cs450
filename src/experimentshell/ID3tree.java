/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package decisiontree;
import java.util.ArrayList;
import java.util.Arrays;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
/**
 *
 * @author Harvey
 */
public class ID3tree extends Classifier{
    Instances mInstances;
    Instances mSubset;
    int mNumInst;
    
    /**
     *
     * @param inst
     * @throws Exception
     */
    @Override
     public void buildClassifier(Instances inst) throws Exception {
        mInstances = inst;
        mNumInst = inst.numInstances();
        int splitNum;
        splitNum = calculateSplit(mInstances);
    }   
     
    /**
     *
     * @param instance
     * @return
     * @throws Exception
     */
    @Override
     public double classifyInstance(Instance instance) throws Exception {
        return 2;        
    }
    
    private int calculateSplit(Instances inst)
    {
        Instance tempInst;
        ArrayList<Instance> subset;
        subset = new ArrayList();
        double[] entropy;
        int numAttr = inst.numAttributes();
        entropy = new double[numAttr];
        double tempEnt;
        int numInst = inst.numInstances();
        int splitVal = 5;
        
             
        for (int i = 0; i < numInst; i++)
        {
            tempInst = inst.instance(i);
            subset.add(tempInst);
        }
        
        for (int j = 0; j < numAttr - 1; j++)
        {
            tempEnt = calculateEntropy(subset, numAttr, j);
            entropy[j] = tempEnt;
        }
       double temp = 5.0;
        for (int k = 0; k < numAttr - 1; k++)
        {
            if (temp > entropy[k])
            {
                temp = entropy[k];
                splitVal = k;
            }
        }
        
        return splitVal;
    };
    
    private double calculateEntropy(ArrayList<Instance> aList, int attr, int index) {
        Instance tempInst;
        int numInst = aList.size();
        int[] numAttr;
        double ent = 0.0;
        double temp;
        numAttr = new int[attr];
        Arrays.fill(numAttr, 0);
        
        // Count number of classes         
        for (int i = 0; i < numInst; i++)
        {
            tempInst = aList.get(i);
            for (int j = 0; j < attr - 1; j++)
            {
                if ( tempInst.value(index) == j)
                    numAttr[j] += 1;
            }
        }
        
        //System.out.println(attr);
        for (int j = 0; j < attr - 1; j++)
        {
            temp = numAttr[j] / (double) numInst;
            
            if (Double.isNaN(temp))
                ent -= 0.0;
            else
                ent -= (temp * (Math.log(temp)/Math.log(2)));
        }              

        return ent;        
    };
    
}
