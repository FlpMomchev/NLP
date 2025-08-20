# Beispieltext, der passen sollte:
PASS_TEXT = """
When the Customer places an order, the System checks inventory.
If the item is in stock, the System processes payment.
Once paid, the System sends a confirmation email.
"""

# Beispieltext, bei dem die Akteurs-/Aktionsdichte (Akteure fehlen) und Signale ausfallen sollten:
FAIL_TEXT = """
Place order and check inventory.
Process payment.
"""

REZEPT = """
Steps:

First, preheat your oven to 190°C and lightly grease a baking sheet.
In a large bowl, thoroughly cream together the butter, granulated sugar, and brown sugar using an electric mixer.
Next, beat in the eggs one at a time, followed by the vanilla extract.
In a separate bowl, combine the flour, salt, and baking soda. Gradually add this mixture to the butter mixture, stirring well.
Gently fold in the chocolate chips.
Using a spoon, drop even portions of dough onto the prepared baking sheet.
Bake for 9–11 minutes or until golden brown. Meanwhile, set up a cooling rack.
Carefully remove the cookies from the oven and allow them to cool for a minute on the baking sheet.
Finally, transfer the cookies to the cooling rack to cool completely.
"""

BPMN_PROZESS = """
Let’s say we want to model a process with concurring instances. We are using a simple example. If one credit check of a customer is running, we do not want another credit check for the same customer to be performed at the same time.
The reason could be that the total number of credit checks performed influences the result of the check.
Let’s assume that we are running a credit check for a customer and we get a second request for the same customer at the same time.
What all solutions have in common is that every new instance needs to check for concurring instances on the data level before starting the actual credit check."""
