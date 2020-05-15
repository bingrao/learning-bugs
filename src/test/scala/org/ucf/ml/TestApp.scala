package org.ucf.ml
/**
  * @author
  */
import org.junit.Test
import org.junit.Assert._

class ScalaTestAPP {
  @Test def testAdd() {
    val inputClass =
      """
        |package org.ucf.ml;
        |
        |import java.util.Arrays;
        |import java.util.List;
        |import org.apache.commons.lang3.StringUtils;
        |
        |public class JavaApp {
        |
        |    public void hello(String input) {
        |        List<String> messages = Arrays.asList("hello", "baeldung", "readers!");
        |        messages.forEach(word -> StringUtils.capitalize(word));
        |        messages.forEach(StringUtils::capitalize);
        |
        |        List<Integer> numbers = Arrays.asList(5, 3, 50, 24, 40, 2, 9, 18);
        |        numbers.stream().sorted((a, b) -> a.compareTo(b));
        |        numbers.stream().sorted(Integer::compareTo);
        |    }
        |}
        |""".stripMargin

    val cu = TestUtils.getComplationUnit(inputClass, CLASS, false)
    TestUtils.printAST(outPath=null, cu = cu, format = "ymal")
  }

}