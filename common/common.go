package common

import (
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"reflect"
)

// init creates the interface map to allow packages to add to the register
func init() {
	interfaceMap = make(map[string]interface{})
}

// DoesNotMatch is an error returned if the test marshal and unmarshal
// does not pass
var DoesNotMatch error = errors.New("Does not match")

// InterfaceTestMarshalAndUnmarshal tests that the type in the interface
// is correctly marshaled and unmarshaled. Returns an error if they do
// not match
func InterfaceTestMarshalAndUnmarshal(l interface{}) error {
	l1 := &InterfaceMarshaler{I: l}
	b, err := l1.MarshalJSON()
	if err != nil {
		return err
	}
	l2 := &InterfaceMarshaler{}
	err = l2.UnmarshalJSON(b)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(l, l2.I) {
		return DoesNotMatch
	}
	return nil
}

// interfaceMap helps encoding and decoding interfaces
var interfaceMap map[string]interface{}

// registerString converts the input type to the string
// key in interfaceMap
func registerString(i interface{}) string {
	str := interfaceFullTypename(i)
	// See if it's a pointer, and if so append a * at the end
	if reflect.ValueOf(i).Kind() == reflect.Ptr {
		str += "*"
	}

	return str
}

// Register logs an underlying type to allow encoding and decoding
// as a value of an interface with InterfaceMarshaler. Usually, types
// will be registered in an init() function of a package. This follows
// in spirit with encoding/gob, though actual behavior may vary.
// Like gob, this function will panic if the type is already registered.
// For the purposes of this package, a type and a pointer to a type are
// considered as different functions.
//
// If there is ever a generic encoding/decoding package in the standard
// library that handles interfaces, this will be replaced.
func Register(i interface{}) {
	// Get the string of the type
	str := registerString(i)
	// Check if already registered
	_, ok := interfaceMap[str]
	if ok {
		panic("common/Register: type " + str + " already registered")
	}

	// Extract the real underlying type, make a copy of it, re-cast
	// as an interface{}, and save the interface in the interfaceMap
	// If the kind is a pointer, still save the real value
	isPtr := reflect.ValueOf(i).Kind() == reflect.Ptr
	var newVal interface{}
	var tmp interface{}
	if isPtr {
		tmp = reflect.ValueOf(i).Elem().Interface()
	} else {
		tmp = i
	}
	newVal = reflect.New(reflect.TypeOf(tmp)).Elem().Interface()

	// Either way, save a real value

	//TODO: Add in something where the types aren't copied twice for the *
	interfaceMap[str] = newVal
}

// NotRegistered is retured if the type is not registered
type NotRegistered struct {
	Type string
}

func (n *NotRegistered) Error() string {
	return fmt.Sprintf("common: type %s not registered", n.Type)
}

func ptrFromString(str string) (interface{}, bool, error) {
	val, ok := interfaceMap[str]
	if !ok {
		return nil, false, &NotRegistered{Type: str}
	}
	isPtr := str[len(str)-1:len(str)] == "*"

	return reflect.New(reflect.TypeOf(val)).Interface(), isPtr, nil
}

/*
// fromString returns a copy of the registered type
func fromString(str string) (interface{}, error) {
	val, isPtr, err := ptrFromString(str)
	if err != nil {
		return nil, err
	}
	var newVal interface{}
	newVal = val
	if !isPtr {
		newVal = reflect.ValueOf(val).Elem().Interface()
	}
	return newVal, nil
}
*/

// NotInPackage is an error which signifies the type is
// not in the package. Normally used with marshaling and
// unmarshaling
var NotInPackage = errors.New("NotInPackage")

// UnmarshalMismatch is an error type used when unmarshaling the specific
// activators in this package.
type UnmarshalMismatch struct {
	Expected string
	Received string
}

// Error is so UnmarshalMismatch implements the error interface
func (u UnmarshalMismatch) Error() string {
	return "Unmarshal string mismatch. Expected: " + u.Expected + " Received: " + u.Received
}

// InterfaceMarshaler is a type to help the marshaling and unmarshaling
// of interface values. Types marshaled and unmarshaled with InterfaceMarshaler
// must be first be registered using Register(). It uses a similar idea
// to gob.
//
// If there is ever a generic encoding/decoding package in the standard
// library that handles interfaces, this will be replaced.
type InterfaceMarshaler struct {
	I interface{}
}

type lossMarshaler struct {
	Type  string
	Value interface{}
}

type typeUnmarshaler struct {
	Type  string
	Value json.RawMessage
}

func (l InterfaceMarshaler) MarshalJSON() ([]byte, error) {
	// Check that the type has been registered
	name := registerString(l.I)
	_, ok := interfaceMap[name]
	if !ok {
		return nil, &NotRegistered{Type: name}
	}
	marshaler := &lossMarshaler{
		Type:  name,
		Value: l.I,
	}
	return json.Marshal(marshaler)
}
func (l *InterfaceMarshaler) UnmarshalJSON(data []byte) error {
	// First, unmarshal the type
	t := &typeUnmarshaler{}
	err := json.Unmarshal(data, t)
	if err != nil {
		return err
	}
	// Get a pointer to the right type
	val, isPtr, err := ptrFromString(t.Type)
	if err != nil {
		return errors.New("nnet/common error unmarshaling: " + err.Error())
	}

	nextdata := []byte(t.Value)
	// Unmarshal the (pointer to the) value
	err = json.Unmarshal(nextdata, val)
	if err != nil {
		return err
	}

	// If we don't want an interface, return a pointer to it
	if !isPtr {
		val = reflect.ValueOf(val).Elem().Interface()
	}
	l.I = val
	return nil
}

// interfaceFullTypename returns the full name of the provided type
// as packagename/typename
func interfaceFullTypename(i interface{}) string {
	pkgpath, pkgname := interfaceLocation(i)
	return filepath.Join(pkgpath, pkgname)
}

// interfaceLocation finds the package path and typename of the given
// interface
func interfaceLocation(i interface{}) (pkgpath string, name string) {
	if reflect.ValueOf(i).Kind() == reflect.Ptr {
		pkgpath = reflect.ValueOf(i).Elem().Type().PkgPath()
		name = reflect.ValueOf(i).Elem().Type().Name()
	} else {
		pkgpath = reflect.TypeOf(i).PkgPath()
		name = reflect.TypeOf(i).Name()
	}
	return
}
