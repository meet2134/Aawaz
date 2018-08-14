package com.example.pulkit.opencvtest;

import android.gesture.OrientedBoundingBox;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.Drawable;
import android.media.Image;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.ImageView;
import java.io.*;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;


import java.io.FileReader;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;

public class MainActivity extends AppCompatActivity implements CvCameraViewListener2  {

    // Used for logging success or failure messages
    private static final String TAG = "OCVSample::Activity";
    // These variables are used (at the moment) to fix camera orientation from 270degree to 0degree
    Mat mRgba;
    Mat mRgbaF;
    Mat mRgbaT;

    // Pulkit SInghal
    private Mat mHsvMat;
    private Mat mMaskMat;
    private Mat mDilatedMat;
    private Mat hierarchy;
    private Mat temp;
    private MatOfInt hull ;
    private Mat mask;
    private Mat result;



    private Scalar CONTOUR_COLOR;

    private int channelCount = 3;
    private int iLineThickness = 3;

    private List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
    private List<MatOfPoint> mContours = new ArrayList<MatOfPoint>();
    private List<MatOfPoint> mContours1 = new ArrayList<MatOfPoint>();
    private List<MatOfPoint> mMaxContours = new ArrayList<MatOfPoint>();

    private Scalar colorGreen = new Scalar(0, 255, 0);
    private Scalar colorRed = new Scalar(255, 0, 0);

    private static double mMinContourArea = 0.3;

    // Loads camera view of OpenCV for us to use. This lets us see using OpenCV
    private CameraBridgeViewBase mOpenCvCameraView;
    // Used in Camera selection from menu (when implemented)
    private boolean mIsJavaCamera = true;
    private MenuItem mItemSwitchCamera = null;
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();

                    // Rakesh kumar suthar
                    mHsvMat = new Mat();
                    mMaskMat = new Mat();
                    mDilatedMat = new Mat();
                    hierarchy = new Mat();
                    temp = new Mat();
                    CONTOUR_COLOR = new Scalar(255,0,0,255);
                    hull = new MatOfInt();
                    mask = new Mat();
                    result = new Mat();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public MainActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);

        mOpenCvCameraView = (JavaCameraView) findViewById(R.id.show_camera_activity_java_surface_view);

        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);

        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {

        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mRgbaF = new Mat(height, width, CvType.CV_8UC4);
        mRgbaT = new Mat(width, width, CvType.CV_8UC4);
    }

    public void onCameraViewStopped() {
        mRgba.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {


        mDilatedMat.release();
        // TODO Auto-generated method stub
        mRgba = inputFrame.rgba();
        // Rotate mRgba 90 degrees
        Core.transpose(mRgba, mRgbaT);
        Imgproc.resize(mRgbaT, mRgbaF, mRgbaF.size(), 0, 0, 0);
        Core.flip(mRgbaF, mRgba, 1);

        // Pulkit Singhal

        Scalar lowerThreshold = new Scalar(0, 48, 80); // skin color – lower hsv values
        Scalar upperThreshold = new Scalar(20, 255, 255); // skin color – higher hsv values

        Imgproc.cvtColor(mRgba,mHsvMat,Imgproc.COLOR_RGB2HSV);
        Core.inRange(mHsvMat,lowerThreshold,upperThreshold,mMaskMat);

        org.opencv.core.Size s1 = new org.opencv.core.Size(11,11);
        org.opencv.core.Size s2 = new org.opencv.core.Size(3,3);
        final org.opencv.core.Point p = new org.opencv.core.Point(-1,-1);
        Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE,s1, p);

        Imgproc.erode(mMaskMat,mMaskMat,new Mat());
        Imgproc.erode(mMaskMat,mMaskMat,new Mat());
        Imgproc.erode(mMaskMat,mMaskMat,new Mat());

        Imgproc.dilate(mMaskMat,mMaskMat,new Mat());
        Imgproc.dilate(mMaskMat,mMaskMat,new Mat());
        Imgproc.dilate(mMaskMat,mMaskMat,new Mat());

        Imgproc.GaussianBlur(mMaskMat,mMaskMat,s2,1,1);

        Core.bitwise_and(mRgba,mRgba,mDilatedMat,mMaskMat);

        // Black and White
        Imgproc.threshold(mDilatedMat,mDilatedMat,20,250,Imgproc.THRESH_BINARY);

        Imgproc.cvtColor(mDilatedMat,mDilatedMat,Imgproc.COLOR_RGB2GRAY);

        final List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(mDilatedMat,contours,hierarchy,Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);

        Imgproc.cvtColor(mDilatedMat,mDilatedMat,Imgproc.COLOR_GRAY2RGB);



        double maxArea = 0;
        MatOfPoint max_contour = new MatOfPoint();
        Iterator<MatOfPoint> iterator = contours.iterator();

        while (iterator.hasNext()){
            MatOfPoint contour = iterator.next();

            double area = Imgproc.contourArea(contour);
            if(area > maxArea){
                maxArea = area;
                max_contour = contour;

            }
        }

        Imgproc.drawContours(mDilatedMat,contours,-1,colorGreen,iLineThickness);




        org.opencv.core.Rect rect = Imgproc.boundingRect(max_contour);



        Imgproc.rectangle(mDilatedMat, rect.tl(), rect.br(), new Scalar(0, 255, 0), 3);
        mask = mDilatedMat.clone();
        MatOfPoint m = new MatOfPoint(new Point(rect.x,rect.y),new Point(rect.x + rect.width, rect.y),new Point(rect.x + rect.width, rect.y + rect.height),new Point(rect.x , rect.y + rect.height));


        Imgproc.fillConvexPoly(mask,m,new Scalar(0,0,255),Imgproc.LINE_8,0);
        Core.inRange(mask,new Scalar(0,0,250),new Scalar(0,0,255),mask);
        Imgproc.threshold(mask,mask,230,255,Imgproc.THRESH_BINARY);
        temp=mask.clone();
        Core.bitwise_and(mDilatedMat,mDilatedMat,result,mask);

        Imgproc.cvtColor(result,result,Imgproc.COLOR_RGB2GRAY);
        Imgproc.threshold(result,result,230,255,Imgproc.THRESH_BINARY);


        double[] diff = new double[10];



        //code for gesture 'Yes'
        Mat sample0 = null;
        try {
            sample0= Utils.loadResource(this,R.drawable.sample0);
        }catch(IOException ex) {
            //Do something with the exception
        }

        Imgproc.cvtColor(sample0,sample0,Imgproc.COLOR_RGB2GRAY);

        final List<MatOfPoint> contours0 = new ArrayList<MatOfPoint>();
        Imgproc.findContours(sample0,contours0,hierarchy,Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);

        Imgproc.cvtColor(sample0,sample0,Imgproc.COLOR_GRAY2RGB);


        double maxArea0 = 0;
        MatOfPoint max_contour0 = new MatOfPoint();
        Iterator<MatOfPoint> iterator0 = contours0.iterator();

        while (iterator0.hasNext()){
            MatOfPoint contour0 = iterator0.next();

            double area0 = Imgproc.contourArea(contour0);
            if(area0 > maxArea0){
                maxArea0 = area0;
                max_contour0 = contour0;

            }
        }

        diff[0] = Imgproc.matchShapes(max_contour,max_contour0,Imgproc.CONTOURS_MATCH_I2,0.0);



        //code for gesture 'One'
        Mat sample1 = null;
        try {
            sample1= Utils.loadResource(this,R.drawable.sample1);
        }catch(IOException ex) {
            //Do something with the exception
        }

        Imgproc.cvtColor(sample1,sample1,Imgproc.COLOR_RGB2GRAY);

        final List<MatOfPoint> contours1 = new ArrayList<MatOfPoint>();
        Imgproc.findContours(sample1,contours1,hierarchy,Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);

        Imgproc.cvtColor(sample1,sample1,Imgproc.COLOR_GRAY2RGB);


        double maxArea1 = 0;
        MatOfPoint max_contour1 = new MatOfPoint();
        Iterator<MatOfPoint> iterator1 = contours1.iterator();

        while (iterator1.hasNext()){
            MatOfPoint contour1 = iterator1.next();

            double area1 = Imgproc.contourArea(contour1);
            if(area1 > maxArea1){
                maxArea1 = area1;
                max_contour1 = contour1;

            }
        }

        diff[1] = Imgproc.matchShapes(max_contour,max_contour1,Imgproc.CONTOURS_MATCH_I2,0.0);



        //code for gesture 'Two'
        Mat sample2 = null;
        try {
            sample2= Utils.loadResource(this,R.drawable.sample2);
        }catch(IOException ex) {
            //Do something with the exception
        }

        Imgproc.cvtColor(sample2,sample2,Imgproc.COLOR_RGB2GRAY);

        final List<MatOfPoint> contours2 = new ArrayList<MatOfPoint>();
        Imgproc.findContours(sample2,contours2,hierarchy,Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);

        Imgproc.cvtColor(sample2,sample2,Imgproc.COLOR_GRAY2RGB);


        double maxArea2 = 0;
        MatOfPoint max_contour2 = new MatOfPoint();
        Iterator<MatOfPoint> iterator2 = contours2.iterator();

        while (iterator2.hasNext()){
            MatOfPoint contour2 = iterator2.next();

            double area2 = Imgproc.contourArea(contour2);
            if(area2 > maxArea2){
                maxArea2 = area2;
                max_contour2 = contour2;

            }
        }

        diff[2] = Imgproc.matchShapes(max_contour,max_contour2,Imgproc.CONTOURS_MATCH_I2,0.0);



        //code for gesture 'Three'
        Mat sample3 = null;
        try {
            sample3= Utils.loadResource(this,R.drawable.sample3);
        }catch(IOException ex) {
            //Do something with the exception
        }

        Imgproc.cvtColor(sample3,sample3,Imgproc.COLOR_RGB2GRAY);

        final List<MatOfPoint> contours3 = new ArrayList<MatOfPoint>();
        Imgproc.findContours(sample3,contours3,hierarchy,Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);

        Imgproc.cvtColor(sample3,sample3,Imgproc.COLOR_GRAY2RGB);


        double maxArea3 = 0;
        MatOfPoint max_contour3 = new MatOfPoint();
        Iterator<MatOfPoint> iterator3 = contours3.iterator();

        while (iterator3.hasNext()){
            MatOfPoint contour3 = iterator3.next();

            double area3 = Imgproc.contourArea(contour3);
            if(area3 > maxArea3){
                maxArea3 = area3;
                max_contour3 = contour3;

            }
        }

        diff[3] = Imgproc.matchShapes(max_contour,max_contour3,Imgproc.CONTOURS_MATCH_I2,0.0);


        //code for gesture 'Four'
        Mat sample4 = null;
        try {
            sample4= Utils.loadResource(this,R.drawable.sample4);
        }catch(IOException ex) {
            //Do something with the exception
        }

        Imgproc.cvtColor(sample4,sample4,Imgproc.COLOR_RGB2GRAY);

        final List<MatOfPoint> contours4 = new ArrayList<MatOfPoint>();
        Imgproc.findContours(sample4,contours4,hierarchy,Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);

        Imgproc.cvtColor(sample4,sample4,Imgproc.COLOR_GRAY2RGB);


        double maxArea4 = 0;
        MatOfPoint max_contour4 = new MatOfPoint();
        Iterator<MatOfPoint> iterator4 = contours4.iterator();

        while (iterator4.hasNext()){
            MatOfPoint contour4 = iterator4.next();

            double area4 = Imgproc.contourArea(contour4);
            if(area4 > maxArea4){
                maxArea4 = area4;
                max_contour4 = contour4;

            }
        }

        diff[4] = Imgproc.matchShapes(max_contour,max_contour4,Imgproc.CONTOURS_MATCH_I2,0.0);


        //code for gesture 'Five'
        Mat sample5 = null;
        try {
            sample5= Utils.loadResource(this,R.drawable.sample5);
        }catch(IOException ex) {
            //Do something with the exception
        }

        Imgproc.cvtColor(sample5,sample5,Imgproc.COLOR_RGB2GRAY);

        final List<MatOfPoint> contours5 = new ArrayList<MatOfPoint>();
        Imgproc.findContours(sample5,contours5,hierarchy,Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);

        Imgproc.cvtColor(sample5,sample5,Imgproc.COLOR_GRAY2RGB);


        double maxArea5 = 0;
        MatOfPoint max_contour5 = new MatOfPoint();
        Iterator<MatOfPoint> iterator5 = contours5.iterator();

        while (iterator5.hasNext()){
            MatOfPoint contour5 = iterator5.next();

            double area5 = Imgproc.contourArea(contour5);
            if(area5 > maxArea5){
                maxArea5 = area5;
                max_contour5 = contour5;

            }
        }

        diff[5] = Imgproc.matchShapes(max_contour,max_contour5,Imgproc.CONTOURS_MATCH_I2,0.0);



        //code for gesture 'No'
        Mat sample6 = null;
        try {
            sample6= Utils.loadResource(this,R.drawable.no);
        }catch(IOException ex) {
            //Do something with the exception
        }

        Imgproc.cvtColor(sample6,sample6,Imgproc.COLOR_RGB2GRAY);

        final List<MatOfPoint> contours6 = new ArrayList<MatOfPoint>();
        Imgproc.findContours(sample6,contours6,hierarchy,Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);

        Imgproc.cvtColor(sample6,sample6,Imgproc.COLOR_GRAY2RGB);


        double maxArea6 = 0;
        MatOfPoint max_contour6 = new MatOfPoint();
        Iterator<MatOfPoint> iterator6 = contours6.iterator();

        while (iterator6.hasNext()){
            MatOfPoint contour6 = iterator6.next();

            double area6 = Imgproc.contourArea(contour6);
            if(area6 > maxArea6){
                maxArea6 = area6;
                max_contour6 = contour6;

            }
        }

        diff[6] = Imgproc.matchShapes(max_contour,max_contour6,Imgproc.CONTOURS_MATCH_I2,0.0);


        //code for gesture 'hackfest'
        Mat sample7 = null;
        try {
            sample7= Utils.loadResource(this,R.drawable.hackfest);
        }catch(IOException ex) {
            //Do something with the exception
        }

        Imgproc.cvtColor(sample7,sample7,Imgproc.COLOR_RGB2GRAY);

        final List<MatOfPoint> contours7= new ArrayList<MatOfPoint>();
        Imgproc.findContours(sample7,contours7,hierarchy,Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);

        Imgproc.cvtColor(sample7,sample7,Imgproc.COLOR_GRAY2RGB);


        double maxArea7 = 0;
        MatOfPoint max_contour7 = new MatOfPoint();
        Iterator<MatOfPoint> iterator7 = contours7.iterator();

        while (iterator7.hasNext()){
            MatOfPoint contour7 = iterator7.next();

            double area7 = Imgproc.contourArea(contour7);
            if(area7 > maxArea7){
                maxArea7 = area7;
                max_contour7 = contour7;

            }
        }

        diff[7] = Imgproc.matchShapes(max_contour,max_contour7,Imgproc.CONTOURS_MATCH_I2,0.0);


        //code for gesture 'love'
        Mat sample8 = null;
        try {
            sample8= Utils.loadResource(this,R.drawable.love);
        }catch(IOException ex) {
            //Do something with the exception
        }

        Imgproc.cvtColor(sample8,sample8,Imgproc.COLOR_RGB2GRAY);

        final List<MatOfPoint> contours8 = new ArrayList<MatOfPoint>();
        Imgproc.findContours(sample8,contours8,hierarchy,Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);

        Imgproc.cvtColor(sample8,sample8,Imgproc.COLOR_GRAY2RGB);


        double maxArea8 = 0;
        MatOfPoint max_contour8 = new MatOfPoint();
        Iterator<MatOfPoint> iterator8 = contours8.iterator();

        while (iterator8.hasNext()){
            MatOfPoint contour8 = iterator8.next();

            double area8 = Imgproc.contourArea(contour8);
            if(area8 > maxArea8){
                maxArea8 = area8;
                max_contour8 = contour8;

            }
        }

        diff[8] = Imgproc.matchShapes(max_contour,max_contour8,Imgproc.CONTOURS_MATCH_I2,0.0);


        //code for gesture 'stop'
        Mat sample9 = null;
        try {
            sample9= Utils.loadResource(this,R.drawable.stop);
        }catch(IOException ex) {
            //Do something with the exception
        }

        Imgproc.cvtColor(sample9,sample9,Imgproc.COLOR_RGB2GRAY);

        final List<MatOfPoint> contours9 = new ArrayList<MatOfPoint>();
        Imgproc.findContours(sample9,contours9,hierarchy,Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);

        Imgproc.cvtColor(sample9,sample9,Imgproc.COLOR_GRAY2RGB);


        double maxArea9 = 0;
        MatOfPoint max_contour9 = new MatOfPoint();
        Iterator<MatOfPoint> iterator9 = contours9.iterator();

        while (iterator9.hasNext()){
            MatOfPoint contour9 = iterator9.next();

            double area9 = Imgproc.contourArea(contour9);
            if(area9 > maxArea9){
                maxArea9 = area9;
                max_contour9 = contour9;

            }
        }

        diff[9] = Imgproc.matchShapes(max_contour,max_contour9,Imgproc.CONTOURS_MATCH_I2,0.0);






        Imgproc.cvtColor(result,result,Imgproc.COLOR_GRAY2RGB);

        int index=10;
        double min=100;
        for(int i=0;i<10;i++){
            if(diff[i]<min){
                min=diff[i];
                index=i;
            }
        }

        if(diff[index] < 1 && index < 10 && index != 6 && index != 1) {

            org.opencv.core.Rect rect1 = Imgproc.boundingRect(max_contour);

            Mat res = new Mat();
            Core.bitwise_and(result,result,res,temp);
            result = res.clone();
            Imgproc.rectangle(result, rect1.tl(), rect1.br(), new Scalar(0, 255, 0), 3);
            // Adding Text
            if(index==0)
                Imgproc.putText(result,"Gesture Detected : 'OK'",new Point(5,50),Core.FONT_HERSHEY_SIMPLEX,1.1,new Scalar(0,255,0),3);
                //   else if(index==1)
                //     Imgproc.putText(result,"Gesture Detected : 'One'",new Point(5,50),Core.FONT_HERSHEY_SIMPLEX,1.1,new Scalar(0,255,0),3);
            else if(index==2)
                Imgproc.putText(result,"Gesture Detected : 'Two'",new Point(5,50),Core.FONT_HERSHEY_SIMPLEX,1.1,new Scalar(0,255,0),3);
            else if(index==3)
                Imgproc.putText(result,"Gesture Detected : 'Three'",new Point(5,50),Core.FONT_HERSHEY_SIMPLEX,1.1,new Scalar(0,255,0),3);
            else if(index==4)
                Imgproc.putText(result,"Gesture Detected : 'Four'",new Point(5,50),Core.FONT_HERSHEY_SIMPLEX,1.1,new Scalar(0,255,0),3);
            else if(index==5)
                Imgproc.putText(result,"Gesture Detected : 'Five'",new Point(5,50),Core.FONT_HERSHEY_SIMPLEX,1.1,new Scalar(0,255,0),3);
                // else if(index==6)
                //   Imgproc.putText(result,"Gesture Detected : 'No'",new Point(5,50),Core.FONT_HERSHEY_SIMPLEX,1.1,new Scalar(0,255,0),3);
            else if(index==7)
                Imgproc.putText(result,"Gesture Detected : 'Hackfest'",new Point(5,50),Core.FONT_HERSHEY_SIMPLEX,1.1,new Scalar(0,255,0),3);
            else if(index==8)
                Imgproc.putText(result,"Gesture Detected : 'Love'",new Point(5,50),Core.FONT_HERSHEY_SIMPLEX,1.1,new Scalar(0,255,0),3);
            else if(index==9)
                Imgproc.putText(result,"Gesture Detected : 'Stop'",new Point(5,50),Core.FONT_HERSHEY_SIMPLEX,1.1,new Scalar(0,255,0),3);


        }
        else{


            org.opencv.core.Rect rect1 = Imgproc.boundingRect(max_contour);
            Imgproc.rectangle(result, rect1.tl(), rect1.br(), new Scalar(255, 0, 0), 3);

        }


        return result; // This function must return
    }
}