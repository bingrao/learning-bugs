package org.ucf.ml
/**
  * @author
  */
import org.junit.Test
import org.junit.Assert._

class ScalaTestAPP {
  @Test def testAdd() {
    println("Hello World From Scala")
    assertTrue(true)
  }

  @Test def testReadSource() {
    val path = "data/JavaApp.java"
    val srcReader = new plaintext.SourceCodeAnalyzer
    val src = srcReader.readSourceCode(path)
    println(src + "\n")
    println(srcReader.removeCommentsAndAnnotations(src))
  }

  @Test def testParser() {
    val path = "data/JavaApp.java"
//    val p = new parser.Parser
//    val src = p.parseFile(path)
//    println(src)
  }
}