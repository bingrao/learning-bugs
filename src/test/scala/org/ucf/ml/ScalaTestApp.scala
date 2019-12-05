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
    val path = "data/testUnits/JavaApp.java"
    val srcReader = new plaintext.SourceCodeAnalyzer
    val src = srcReader.readSourceCode(path)
    println(src + "\n")
    println(srcReader.removeCommentsAndAnnotations(src))
  }

  @Test def testParser() {
    val path = "data/testUnits/1.f.java"
    val p = new parser.Parser
    p.parseFile(path)
  }
}