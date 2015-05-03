/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experimentshell;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.EuclideanDistance;
import weka.core.Attribute;

/**
 *
 * @author Harvey
 */
public class KnnClassifier extends Classifier {
    Instances mInstances;
    int mClass;
    int mK;
    /**
     *
     * @param i
     * @throws Exception
     */
    @Override
    public void buildClassifier(Instances i) throws Exception {
        mInstances = i;
        mClass = 0;
        //System.out.println(mInstances);
    }
    
    /**
     *
     * @param pInst
     * @return
     * @throws Exception
     */
    @Override
    public double classifyInstance(Instance pInst) throws Exception {
        mK = 3;
        boolean same = true;
        EuclideanDistance eDist = new EuclideanDistance();
        eDist.setInstances(mInstances);
        int numInst = mInstances.numInstances();
        Instance[] KNN;
        KNN = new Instance[mK]; //array of three
        double[] distKNN;
        distKNN = new double[mK];
        
        //System.out.println(eDist.getInstances());
        for (int i = 0; i < numInst; i++)
        {
            Instance tempInst = mInstances.instance(i);
            
            double dist = eDist.distance(tempInst, pInst);
            
            //loop through 3 array so soon as all instancs loops
            // finish the 3 smallest will I have.
            
            for (int j = 0; j < mK; j++)
            {
                if (distKNN[j] == 0.0)
                {
                    distKNN[j] = dist;
                    KNN[j] = tempInst;
                    break;
                }
                else if (distKNN[j] < dist)
                {
                    distKNN[j] = dist;
                    KNN[j] = tempInst;
                }
            }
        }
        
        for (int temp = 0; temp < mK - 1; temp++)
        {
            
            if (KNN[temp] != KNN[temp + 1])
                same = false;
        }
        if (same)
        {
            if(KNN[0].attribute(4).name() == "Iris-setosa")
                mClass = 0;
            else if (KNN[0].attribute(4).name() == "Iris-versicolor")
                mClass = 1;
            else
                mClass = 2;
        }
       
        mClass = KNN[0].attribute(4).type();
        System.out.println(mClass);
        return mClass;        
    }

    
    
}
