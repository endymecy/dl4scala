package org.dl4scala.util


import org.slf4j.LoggerFactory
import org.apache.commons.compress.archivers.tar.TarArchiveEntry
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import org.apache.commons.io.FileUtils
import org.apache.commons.io.IOUtils
import java.io._
import java.util.zip.GZIPInputStream
import java.util.zip.ZipInputStream
/**
  * Created by endy on 2017/8/28.
  */
object ArchiveUtils {

  private val log = LoggerFactory.getLogger(ArchiveUtils.getClass)

  /**
    * Extracts files to the specified destination
    *
    * @param file the file to extract to
    * @param dest the destination directory
    * @throws IOException
    */
  @throws[IOException]
  def unzipFileTo(file: String, dest: String): Unit = {
    val target = new File(file)
    if (!target.exists) throw new IllegalArgumentException("Archive doesnt exist")
    val fin = new FileInputStream(target)
    val BUFFER = 2048
    val data = new Array[Byte](BUFFER)
    if (file.endsWith(".zip")) { //getFromOrigin the zip file content
      val zis = new ZipInputStream(fin)
      //getFromOrigin the zipped file list entry
      var ze = zis.getNextEntry
      while (ze != null) {
        val fileName = ze.getName
        val newFile = new File(dest + File.separator + fileName)
        log.info("file unzip : " + newFile.getAbsoluteFile)
        //createComplex all non exists folders
        //else you will hit FileNotFoundException for compressed folder
        new File(newFile.getParent).mkdirs
        val fos = new FileOutputStream(newFile)
        Stream.continually(zis.read(data)).takeWhile(x => x > 0).foreach{x =>
          fos.write(data, 0, x)
        }

        fos.close()
        ze = zis.getNextEntry
      }
      zis.closeEntry()
      zis.close()
    }
    else if (file.endsWith(".tar.gz") || file.endsWith(".tgz")) {
      val in = new BufferedInputStream(fin)
      val gzIn = new GzipCompressorInputStream(in)
      val tarIn = new TarArchiveInputStream(gzIn)
      var entry: TarArchiveEntry = null
      /** Read the tar entries using the getNextEntry method **/
      while ((entry = tarIn.getNextEntry.asInstanceOf[TarArchiveEntry]) != null) {
        log.info("Extracting: " + entry.getName)
        /** If the entry is a directory, createComplex the directory. **/
        if (entry.isDirectory) {
          val f = new File(dest + File.separator + entry.getName)
          f.mkdirs
        }
        else {
          /**
            * If the entry is a file,write the decompressed file to the disk
            * and close destination stream.
            **/
          var count = 0
          val fos = new FileOutputStream(dest + File.separator + entry.getName)
          val destStream = new BufferedOutputStream(fos, BUFFER)
          while ( {
            (count = tarIn.read(data, 0, BUFFER)) != -1
          }) destStream.write(data, 0, count)
          destStream.flush()
          IOUtils.closeQuietly(destStream)
        }
      }
      /** Close the input stream **/
      tarIn.close()
    }
    else if (file.endsWith(".gz")) {
      val is2 = new GZIPInputStream(fin)
      val extracted = new File(target.getParent, target.getName.replace(".gz", ""))
      if (extracted.exists) extracted.delete
      extracted.createNewFile
      val fos = FileUtils.openOutputStream(extracted)
      IOUtils.copyLarge(is2, fos)
      is2.close()
      fos.flush()
      fos.close()
    }
    target.delete
  }
}
