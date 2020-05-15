package org.ucf.ml
/**
 * @author
 */
import org.junit.Test
import org.junit.Assert._

class TestAbstract {
  @Test def testAbstract() {
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

    println(TestUtils.get_abstract_code(inputClass, CLASS, false))

  }

  @Test def testAbstractFile(): Unit ={
    val input = "data/raw/buggy/1.java"
    println(TestUtils.get_abstract_code(input, METHOD, true))
  }
}
