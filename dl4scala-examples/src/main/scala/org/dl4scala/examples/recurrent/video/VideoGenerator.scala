package org.dl4scala.examples.recurrent.video

import java.awt.geom.{Arc2D, Ellipse2D, Line2D, Rectangle2D}
import java.awt.image.BufferedImage
import java.awt.{BasicStroke, Color, RenderingHints}
import java.io.File
import java.nio.file.{Files, Paths, StandardOpenOption}

import org.apache.commons.io.FilenameUtils
import org.jcodec.api.SequenceEncoder

import scala.util.Random

/**
  * A support class for generating a synthetic video data set
  * Created by endy on 2017/6/14.
  */
object VideoGenerator {
  val NUM_SHAPES = 4 // 0=circle, 1=square, 2=arc, 3=line

  val MAX_VELOCITY = 3
  val SHAPE_SIZE = 25
  val SHAPE_MIN_DIST_FROM_EDGE = 15
  val DISTRACTOR_MIN_DIST_FROM_EDGE = 0
  val LINE_STROKE_WIDTH = 6 // Width of line (line shape only)

  val lineStroke = new BasicStroke(LINE_STROKE_WIDTH)
  val MIN_FRAMES = 10 // Minimum number of frames the target shape to be present

  val MAX_NOISE_VALUE = 0.5f

  @throws(classOf[Exception])
  private def generateVideo(path: String, nFrames: Int, width: Int, height: Int,
                            numShapes: Int, r: Random, backgroundNoise: Boolean,
                            numDistractorsPerFrame: Int): Array[Int] = {
    // First: decide where transitions between one shape and another are
    // val rns = new Array[Double](numShapes)
    var rns = (0 until numShapes).map(i => r.nextDouble())
    val sum = rns.sum
    rns = rns.map(x => x/sum)

    val startFrames: Array[Int] = new Array[Int](numShapes)
    startFrames(0) = 0
    (1 until numShapes).foreach{i =>
      startFrames(i) = (startFrames(i - 1) + MIN_FRAMES + rns(i) * (nFrames - numShapes * MIN_FRAMES)).asInstanceOf[Int]
    }

    // Randomly generate shape positions, velocities, colors, and type
    val shapeTypes = new Array[Int](numShapes)
    val initialX = new Array[Int](numShapes)
    val initialY = new Array[Int](numShapes)
    val velocityX = new Array[Double](numShapes)
    val velocityY = new Array[Double](numShapes)
    val color = new Array[Color](numShapes)

    (0 until numShapes).foreach{i =>
      shapeTypes(i) = r.nextInt(NUM_SHAPES)
      initialX(i) = SHAPE_MIN_DIST_FROM_EDGE + r.nextInt(width - SHAPE_SIZE - 2 * SHAPE_MIN_DIST_FROM_EDGE)
      initialY(i) = SHAPE_MIN_DIST_FROM_EDGE + r.nextInt(height - SHAPE_SIZE - 2 * SHAPE_MIN_DIST_FROM_EDGE)
      velocityX(i) = -1 + 2 * r.nextDouble
      velocityY(i) = -1 + 2 * r.nextDouble
      color(i) = new Color(r.nextFloat, r.nextFloat, r.nextFloat)
    }

    //Generate a sequence of BufferedImages with the given shapes, and write them to the video
    val enc = new SequenceEncoder(new File(path))

    var currShape = 0
    val labels = new Array[Int](nFrames)

    (0 until nFrames).foreach{i =>
      if (currShape < numShapes - 1 && i >= startFrames(currShape + 1)) currShape += 1

      val bi = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
      val g2d = bi.createGraphics
      g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
      g2d.setBackground(Color.BLACK)

      if(backgroundNoise) (0 until width).foreach{ x =>
        (0 until height).foreach{y =>
          bi.setRGB(x, y,
            new Color(r.nextFloat * MAX_NOISE_VALUE, r.nextFloat * MAX_NOISE_VALUE,
              r.nextFloat * MAX_NOISE_VALUE).getRGB)
        }
      }

      g2d.setColor(color(currShape))

      //Position of shape this frame
      val currX = (initialX(currShape) + (i - startFrames(currShape)) * velocityX(currShape) * MAX_VELOCITY).asInstanceOf[Int]
      val currY = (initialY(currShape) + (i - startFrames(currShape)) * velocityY(currShape) * MAX_VELOCITY).asInstanceOf[Int]

      shapeTypes(currShape) match {
        case 0 =>
          //Circle
          g2d.fill(new Ellipse2D.Double(currX, currY, SHAPE_SIZE, SHAPE_SIZE))
        case 1 =>
          //Square
          g2d.fill(new Rectangle2D.Double(currX, currY, SHAPE_SIZE, SHAPE_SIZE))
        case 2 =>
          //Arc
          g2d.fill(new Arc2D.Double(currX,currY,SHAPE_SIZE,SHAPE_SIZE,315,225,Arc2D.PIE))
        case 3 =>
          import java.awt.geom.Line2D
          //Line//Line
          g2d.setStroke(lineStroke)
          g2d.draw(new Line2D.Double(currX, currY, currX + SHAPE_SIZE, currY + SHAPE_SIZE))
        case _ =>
          throw new RuntimeException
      }

      // Add some distractor shapes, which are present for one frame only
      (0 until numDistractorsPerFrame).foreach{j =>
        val distractorShapeIdx = r.nextInt(NUM_SHAPES)

        val distractorX = DISTRACTOR_MIN_DIST_FROM_EDGE + r.nextInt(width - SHAPE_SIZE)
        val distractorY = DISTRACTOR_MIN_DIST_FROM_EDGE + r.nextInt(height - SHAPE_SIZE)
        g2d.setColor(new Color(r.nextFloat, r.nextFloat, r.nextFloat))

        distractorShapeIdx match {
          case 0 =>
            g2d.fill(new Ellipse2D.Double(distractorX, distractorY, SHAPE_SIZE, SHAPE_SIZE))
          case 1 =>
            g2d.fill(new Rectangle2D.Double(distractorX, distractorY, SHAPE_SIZE, SHAPE_SIZE))
          case 2 =>
            g2d.fill(new Arc2D.Double(distractorX, distractorY, SHAPE_SIZE, SHAPE_SIZE, 315, 225, Arc2D.PIE))
          case 3 =>
            g2d.setStroke(lineStroke)
            g2d.draw(new Line2D.Double(distractorX, distractorY, distractorX + SHAPE_SIZE, distractorY + SHAPE_SIZE))
          case _ =>
            throw new RuntimeException
        }

      }
      enc.encodeImage(bi)
      g2d.dispose()
      labels(i) = shapeTypes(currShape)
    }

    enc.finish();   //write .mp4

    labels
  }

  @throws(classOf[Exception])
  def generateVideoData( outputFolder: String,  filePrefix: String, nVideos: Int, nFrames: Int,
    width: Int, height: Int, numShapesPerVideo: Int, backgroundNoise: Boolean,
                         numDistractorsPerFrame: Int, seed: Long): Unit = {
    val r = new Random(seed)
    (0 until nVideos).foreach{i =>
      val videoPath = FilenameUtils.concat(outputFolder, filePrefix + "_" + i + ".mp4")
      val labelsPath = FilenameUtils.concat(outputFolder, filePrefix + "_" + i + ".txt")
      val labels = generateVideo(videoPath, nFrames, width, height, numShapesPerVideo,
        r, backgroundNoise, numDistractorsPerFrame)

      //Write labels to text file
      val sb = new StringBuilder

      labels.indices.foreach{j =>
        sb.append(labels(j))
        if (j == labels.length - 1) sb.append("\n")
      }
      Files.write(Paths.get(labelsPath), sb.toString.getBytes("utf-8"),
        StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
    }
  }
}
