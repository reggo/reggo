package common

import (
	"encoding/json"
	"fmt"
	"reflect"
	"testing"
)

type Foo struct {
	Dave float64
}

type Bar struct {
	Baz int
}

type Int int

type Quux struct {
}

type Quiz struct{}

// tryRegister returns true if register panics
func tryRegister(i interface{}) (b bool) {
	defer func() {
		if r := recover(); r != nil {
			b = true
		}
	}()
	Register(i)
	return
}

func TestRegisterAndFromString(t *testing.T) {
	// Clear the map
	interfaceMap = make(map[string]interface{})
	// Registering a type should work
	Register(Bar{})
	Register(&Foo{})
	// Also registering a pointer to that type should work
	Register(&Bar{})
	i := Int(0)
	Register(i)

	if !tryRegister(Bar{}) {
		t.Errorf("Should panic when registering the same type twice")
	}
}

func TestMarshal(t *testing.T) {
	interfaceMap = make(map[string]interface{})
	Register(Bar{})
	Register(&Bar{})
	i := Int(0)
	Register(i)
	var err error

	err = testMarshal(Bar{6})
	if err != nil {
		t.Errorf("non-pointer struct: " + err.Error())
	}

	err = testMarshal(&Bar{6})
	if err != nil {
		t.Errorf("pointer struct: " + err.Error())
	}

	err = testMarshal(Int(15))
	if err != nil {
		t.Errorf("integer value: " + err.Error())
	}

}

func TestInterfaceTest(t *testing.T) {
	err := InterfaceTestMarshalAndUnmarshal(Quiz{})
	if err == nil {
		t.Error("Succeeds when type is not registered")
	}
	Register(Quiz{})
	err = InterfaceTestMarshalAndUnmarshal(Quiz{})
	if err != nil {
		t.Error("Fails when type is registered")
	}
}

func testMarshal(i interface{}) error {
	v := &InterfaceMarshaler{I: i}
	b, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("error marshaling: " + err.Error())
	}

	v2 := &InterfaceMarshaler{}
	err = json.Unmarshal(b, v2)
	if err != nil {
		return fmt.Errorf("error unmarshaling" + err.Error())
	}

	if !reflect.DeepEqual(v.I, v2.I) {
		fmt.Println("doesn't match")
		fmt.Printf("%#v\n", v.I)
		fmt.Printf("%#v\n", v2.I)
		return DoesNotMatch
	}
	return nil
}
