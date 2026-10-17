
class Help:
    methods = {}

    def help(self, method=None):
        if method is None:
            print(f"\nHelp for {self.__class__.__name__}")
            print("-" * 40)

            for name, info in self.methods.items():
                print(f"{name} - {info['description']}")

            print("\nUse object.help('method_name') for details.")
            return

        method = method.removesuffix("()")

        if method not in self.methods:
            print(f"Method '{method}' not found.")
            print("Use help() to see available methods.")
            return

        info = self.methods[method]

        print(f"\nMethod: {method}")
        print(f"Description: {info['description']}")
        print(f"Parameters: {info['parameters']}")
        print(f"Returns: {info['returns']}")
        print(f"Example: {info['example']}")
