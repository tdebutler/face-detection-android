package com.debutler.bfacedetectoropencv;

import android.annotation.SuppressLint;
import android.annotation.TargetApi;
import android.content.ClipData;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.support.annotation.Nullable;
import android.support.annotation.RequiresApi;
import android.support.v7.app.AppCompatActivity;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import androidx.exifinterface.media.ExifInterface;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";

    private boolean areImagesReady = false;
    private CascadeClassifier cascade;
    private TextView imagesTotalNumber, facesTotalNumber, showBoundingBoxes, showProgressInfo;
    private final int CODE_MULTIPLE_LOAD = 1;
    private ArrayList<Mat> mMats = new ArrayList<>();
    private ArrayList<String> mPaths = new ArrayList<>();
    private int totalFacesN = 0;

    /**
     * Loads OpenCV library and cascade classifier model at startup.
     */
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            if (status == LoaderCallbackInterface.SUCCESS) {
                Log.i("tag", "OpenCV loaded successfully");
                try {
                    InputStream is = getResources().openRawResource(R.raw.haarcascade_frontalface_alt);
                    File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                    File mCascadeFile = new File(cascadeDir, "cascade.xml");
                    FileOutputStream os = new FileOutputStream(mCascadeFile);

                    byte[] buffer = new byte[4096];
                    int bytesRead;
                    while ((bytesRead = is.read(buffer)) != -1) {
                        os.write(buffer, 0, bytesRead);
                    }
                    is.close();
                    os.close();

                    cascade = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                    if (cascade.empty()) {
                        Log.i("Cascade Error", "Failed to load cascade classifier");
                        cascade = null;
                    }
                } catch (Exception e) {
                    Log.i("Cascade Error: ", "Cascade not found");
                }
                /* default:
                {
                    super.onManagerConnected(status);
                } break; */
            }
        }
    };

    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d("OpenCV", "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d("OpenCV", "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imagesTotalNumber = findViewById(R.id.imgNbTextView);
        facesTotalNumber = findViewById(R.id.facesNbTextView);
        showProgressInfo = findViewById(R.id.progress_info);
        showBoundingBoxes = findViewById(R.id.bbxTextView);
        showBoundingBoxes.setMovementMethod(new ScrollingMovementMethod());
    }

    /**
     * Inflates the menu to allow loading of images
     */
    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }


    /**
     * Launches onActivityResult if OPEN GALLERY is selected.
     */
    @TargetApi(Build.VERSION_CODES.JELLY_BEAN_MR2)
    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        int id = item.getItemId();

        if (id == R.id.open_gallery) {
            Intent intent =new Intent(Intent.ACTION_PICK);
            intent.setType("image/*");
            intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
            startActivityForResult(Intent.createChooser(intent, "Load multiple images"), CODE_MULTIPLE_LOAD);
            return true;
        }
        return super.onOptionsItemSelected(item);
    }


    /**
     * Loads selected images from gallery and pre-processes them.
     */
    @SuppressLint("SetTextI18n")
    @RequiresApi(api = Build.VERSION_CODES.JELLY_BEAN)
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == CODE_MULTIPLE_LOAD && resultCode == RESULT_OK && null != data) {
            ClipData clipData = data.getClipData();

            if (clipData != null) {

                for (int i = 0; i < clipData.getItemCount(); i++) {

                    long startTime = System.nanoTime(); // profiling purposes.

                    ClipData.Item item = clipData.getItemAt(i);
                    Uri uri = item.getUri();
                    Log.i("PICTURE URI: ", uri.toString());

                    String picturePath = getRealPathFromURI(uri);

                    Log.i("PICTURE PATH: ", picturePath);

                    //To speed up loading of image.
                    BitmapFactory.Options options = new BitmapFactory.Options();
                    options.inSampleSize = 1; // if > 1 will sub-sample the image
                    // TODO : sub sample images only if too large. Do the same in Face Clustering App !

                    Bitmap temp = BitmapFactory.decodeFile(picturePath, options);

                    //Get orientation information.
                    int orientation = 0;
                    try {
                        ExifInterface imgParams = new ExifInterface(picturePath);
                        orientation = imgParams.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_UNDEFINED);

                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                    //Rotating the image to get the correct orientation.
                    Matrix rotate90 = new Matrix();
                    rotate90.postRotate(orientation);
                    Bitmap originalBitmap = rotateBitmap(temp, orientation);

                    //Convert Bitmap to Mat.
                    Bitmap tempBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true);
                    Mat originalMat = new Mat(tempBitmap.getHeight(), tempBitmap.getWidth(), CvType.CV_8U);

                    Utils.bitmapToMat(tempBitmap, originalMat);

                    mMats.add(originalMat);
                    mPaths.add(picturePath);

                    long elapsedTime = System.nanoTime() - startTime;
                    Log.i("LOADING IMAGES", "Loading image took " + elapsedTime/1000000 + " ms.");
                }
                areImagesReady = true;
                imagesTotalNumber.setText("Total number of images : " + mMats.size());
            }
        }
    }

//    // to display the current Bitmap
//    private void loadImageToImageView() {
//        ImageView imgView = (ImageView) findViewById(R.id.image_view);
//        imgView.setImageBitmap(currentBitmap);
//    }


    /** Rotate bitmap according to image parameters */
    public static Bitmap rotateBitmap(Bitmap bitmap, int orientation) {

        Matrix matrix = new Matrix();
        switch (orientation) {
            case ExifInterface.ORIENTATION_NORMAL:
                return bitmap;
            case ExifInterface.ORIENTATION_FLIP_HORIZONTAL:
                matrix.setScale(-1, 1);
                break;
            case ExifInterface.ORIENTATION_ROTATE_180:
                matrix.setRotate(180);
                break;
            case ExifInterface.ORIENTATION_FLIP_VERTICAL:
                matrix.setRotate(180);
                matrix.postScale(-1, 1);
                break;
            case ExifInterface.ORIENTATION_TRANSPOSE:
                matrix.setRotate(90);
                matrix.postScale(-1, 1);
                break;
            case ExifInterface.ORIENTATION_ROTATE_90:
                matrix.setRotate(90);
                break;
            case ExifInterface.ORIENTATION_TRANSVERSE:
                matrix.setRotate(-90);
                matrix.postScale(-1, 1);
                break;
            case ExifInterface.ORIENTATION_ROTATE_270:
                matrix.setRotate(-90);
                break;
            default:
                return bitmap;
        }
        try {
            Bitmap bmRotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
            bitmap.recycle();
            return bmRotated;
        } catch (OutOfMemoryError e) {
            e.printStackTrace();
            return null;
        }
    }


    /**
     * Detects faces on a gray scale Mat and stores bounding boxes in rectangles array.
     * Saves all data (bounding boxes & images paths) in a txt file to be later shared for clustering purposes.
     *
     * This function is launched as an async task by the launchGenerateBoundingBoxesAsync function.
     */
    @SuppressLint("SetTextI18n")
    @RequiresApi(api = Build.VERSION_CODES.KITKAT)
    public void generateBoundingBoxes() {
        Log.i("GENERATE BBX", "Running generateBoundingBoxes function");

        if (areImagesReady) {
            long startTime = System.nanoTime(); // profiling purposes

            try {
                FileOutputStream fileOut = openFileOutput("bounding-boxes.txt", MODE_PRIVATE);
                OutputStreamWriter outputStreamWriter = new OutputStreamWriter(fileOut);
                for (int i = 0; i < mMats.size(); i++) { // looping over the images
                    Log.i("GENERATE BBX", "Looping over image no. " + i);

                    final int finalI = i;
                    runOnUiThread(() -> showProgressInfo.setText("Processing image " + (finalI+1) + " out of " + mMats.size()));

                    // converting to gray scale
                    Mat grayMat = new Mat(mMats.get(i).height(), mMats.get(i).width(), CvType.CV_8U);
                    Imgproc.cvtColor(mMats.get(i), grayMat, Imgproc.COLOR_RGB2GRAY);

                    MatOfRect faces = new MatOfRect();
                    if (cascade != null) {
                        // use of cascade classifier model
                        cascade.detectMultiScale(grayMat, faces, 1.1, 3, 2,
                                new Size(2, 2), new Size());
                    }
                    Rect[] facesArray = faces.toArray();

                    for (int j = 0; j < facesArray.length; j++) { // looping over the faces in the image i
                        Log.i("GENERATE BBX", "Looping over face no. " + j);
                        totalFacesN++;
                        Imgproc.rectangle(mMats.get(i), facesArray[j].tl(), facesArray[j].br(),
                                new Scalar(320), 3);

                        // path/--.jpg x-topleft y-topleft x-bottomright y-bottomright
                        outputStreamWriter.write('\n' + mPaths.get(i).replaceAll(" ", "%20") + " "
                                + ((int) facesArray[j].tl().x) + " "
                                + ((int) facesArray[j].tl().y) + " "
                                + ((int) facesArray[j].br().x) + " "
                                + ((int) facesArray[j].br().y));
                        Log.i("GENERATE BBX", "Adding\n" + mPaths.get(i).replaceAll(" ", "%20") + " "
                                + ((int) facesArray[j].tl().x) + " "
                                + ((int) facesArray[j].tl().y) + " "
                                + ((int) facesArray[j].br().x) + " "
                                + ((int) facesArray[j].br().y));
                    }
                    grayMat.release();
                }
                outputStreamWriter.close();
                runOnUiThread(() -> facesTotalNumber.setText("Total number of faces detected : " + totalFacesN));
                runOnUiThread(() -> Toast.makeText(getBaseContext(), "File saved successfully", Toast.LENGTH_SHORT).show());
            } catch (IOException e) {
                e.printStackTrace();
            }

            // reading of the file to display it onscreen
            try {
                FileInputStream fileInputStream = openFileInput("bounding-boxes.txt");
                InputStreamReader inputStreamReader = new InputStreamReader(fileInputStream);

                int READ_BLOCK_SIZE = 100;
                char[] inputBuffer = new char[READ_BLOCK_SIZE];
                StringBuilder s = new StringBuilder();
                int charRead;

                while ((charRead = inputStreamReader.read(inputBuffer)) > 0) {
                    String readString = String.copyValueOf(inputBuffer, 0, charRead);
                    s.append(readString);
                }
                inputStreamReader.close();
                runOnUiThread(() -> showBoundingBoxes.setText(s.toString()));
            } catch (IOException e) {
                e.printStackTrace();
            }

            long elapsedTime = System.nanoTime() - startTime;
            Log.i("GENERATE BBX", "Generating bounding boxes took in total " + elapsedTime / 1000000 + " ms.");
            Log.i("GENERATE BBX", "Took on average " + elapsedTime / (1000000 * mMats.size()) + " ms per image");
            runOnUiThread(() -> showProgressInfo.setText("Generating bounding boxes took on average " + elapsedTime / (1000000 * mMats.size()) + " ms per image"));
        } else {
            Toast.makeText(getBaseContext(), "No images have been selected.", Toast.LENGTH_LONG).show();
        }
    }


    /**
     * To share the bounding boxes data txt file to another app (expected to be Face Clustering App)
     */
    public void shareBoundingBoxes(View v) {
        Intent sendIntent = new Intent();
        sendIntent.setAction(Intent.ACTION_SEND);
        StringBuilder s = new StringBuilder();
        try {
            FileInputStream fileInputStream = openFileInput("bounding-boxes.txt");
            InputStreamReader inputStreamReader = new InputStreamReader(fileInputStream);
            int READ_BLOCK_SIZE = 100;
            char[] inputBuffer = new char[READ_BLOCK_SIZE];

            int charRead;
            Log.i("READING", "bounding-boxes.txt");
            while ((charRead = inputStreamReader.read(inputBuffer)) > 0) {
                String readString = String.copyValueOf(inputBuffer, 0, charRead);
                s.append(readString);
            }
            inputStreamReader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        sendIntent.putExtra(Intent.EXTRA_TEXT, s.toString());
        sendIntent.setType("text/plain");
        startActivity(Intent.createChooser(sendIntent, getResources().getText(R.string.send_to)));
    }


    /**
     * Used to get the real path of an image.
     */
    private String getRealPathFromURI(Uri contentURI) {
        String result;
        Cursor cursor = getContentResolver().query(contentURI, null, null, null, null);
        if (cursor == null) { // Source is Dropbox or other similar local file path
            result = contentURI.getPath();
        } else {
            cursor.moveToFirst();
            int idx = cursor.getColumnIndex(MediaStore.Images.ImageColumns.DATA);
            result = cursor.getString(idx);
            cursor.close();
        }
        return result;
    }


    /**
     * Launches the generateBoundingBoxes function as an Async Task to allow
     * the dynamic displaying of info on the UI.
     */
    public void launchGenerateBoundingBoxesAsync(View view) {
        new GenerateBoundingBoxesAsync().execute();
    }
    private class GenerateBoundingBoxesAsync extends AsyncTask<Void, Void, Void>{
        @androidx.annotation.RequiresApi(api = Build.VERSION_CODES.KITKAT)
        @Override
        protected Void doInBackground(Void... params) {
            generateBoundingBoxes();
            return null;
        }
    }




//    public Mat resizeIfTooBig(Mat originalMat){
//        // Resize images when too big.
//        int width = originalMat.width();
//        int height = originalMat.height();
//        int threshold;
//        if (width < height) { threshold = width; } else { threshold = height; }
//        Mat resizedMat = new Mat();
//        Size sz = new Size(width, height);
//        if (threshold > 5000) {
//            sz = new Size(0.8*width, 0.8*height);
//        } else if (threshold > 4000) {
//            sz = new Size(0.6*width, 0.6*height);
//        } else if (threshold > 3000) {
//            sz = new Size(0.4*width, 0.4*height);
//        } else if (threshold > 2000) {
//            sz = new Size(0.2*width, 0.2*height);
//        }
//        Imgproc.resize(originalMat, resizedMat, sz);
//        return resizedMat;
//    }
}

