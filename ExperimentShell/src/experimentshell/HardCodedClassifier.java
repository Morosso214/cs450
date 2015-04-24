/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experimentshell;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Harvey
 */
public class HardCodedClassifier extends Classifier {
    
    /**
     *
     * @param i
     * @throws Exception
     */
    @Override
    public void buildClassifier(Instances i) throws Exception {
        //Do nothing
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
    
}
