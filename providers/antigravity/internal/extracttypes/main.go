// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Command extracttypes prints the struct types, with field tags, embedded in a stripped Go binary.
//
// It recovers the stream-json DTOs from the agy binary:
//
//	go run ./providers/antigravity/internal/extracttypes ~/.local/bin/agy
//
// GoReSym v3.4.1 cannot read agy: agy is an externally linked PIE whose
// moduledata pointers exist only as R_X86_64_RELATIVE relocations, and it is
// built with Go 1.27+, which replaced moduledata.typelinks with a contiguous
// run of type descriptors. This tool applies the relocations in memory and
// walks that run like runtime.moduleTypelinks.
//
// Only linux/amd64 binaries built with the Go 1.27 internal/abi layouts are
// supported. A layout mismatch stops the walk with an error.
package main

import (
	"debug/elf"
	"encoding/binary"
	"errors"
	"flag"
	"fmt"
	"os"
	"regexp"
	"slices"
	"strings"
)

func main() {
	filter := flag.String("re", `^(printmode|steps)\.|^entrypoints\.modelJSON$`, "regexp matched against struct type names")
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: extracttypes [-re regexp] <binary>")
		os.Exit(2)
	}
	re, err := regexp.Compile(*filter)
	if err == nil {
		err = run(flag.Arg(0), re)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "extracttypes:", err)
		os.Exit(1)
	}
}

func run(path string, re *regexp.Regexp) error {
	m, err := load(path)
	if err != nil {
		return err
	}
	all, err := m.walk()
	if err != nil {
		return err
	}
	for _, t := range m.structs(all, re) {
		m.print(t)
	}
	return nil
}

// Kinds from internal/abi.
const (
	kindComplex128    = 16
	kindArray         = 17
	kindChan          = 18
	kindFunc          = 19
	kindInterface     = 20
	kindMap           = 21
	kindPointer       = 22
	kindSlice         = 23
	kindString        = 24
	kindStruct        = 25
	kindUnsafePointer = 26
)

// TFlag bits from internal/abi.
const (
	tflagUncommon  = 1 << 0
	tflagExtraStar = 1 << 1
	tflagNamed     = 1 << 2
)

// amd64 sizes of the internal/abi descriptor structs in Go 1.27.
const (
	sizeType      = 48              // abi.Type
	sizeUncommon  = 16              // abi.UncommonType
	sizeMethod    = 16              // abi.Method
	sizeArray     = sizeType + 24   // Elem, Slice, Len
	sizeChan      = sizeType + 16   // Elem, Dir
	sizeFunc      = sizeType + 8    // InCount, OutCount, padding
	sizeInterface = sizeType + 32   // PkgPath, Methods
	sizeMap       = sizeType + 88   // Key, Elem, Group, Hasher, 6 uintptr, Flags
	sizePtr       = sizeType + 8    // Elem
	sizeSlice     = sizeType + 8    // Elem
	sizeStruct    = sizeType + 32   // PkgPath, Fields
	sizeField     = 24              // abi.StructField
	sizeImethod   = 8               // abi.Imethod
	rRelative     = 8               // R_X86_64_RELATIVE
	modTypes      = 296             // moduledata.types offset
	modTypedesc   = modTypes + 8    // moduledata.typedesclen offset
	modModuleSize = modTypedesc + 8 // moduledata prefix the tool reads
)

// rtype is a decoded abi.Type header.
type rtype struct {
	va    uint64
	kind  uint8
	tflag uint8
	str   string
}

// image is the binary's loaded sections with relative relocations applied.
type image struct {
	secs        []*elf.Section
	data        [][]byte
	types       uint64
	typedesclen uint64
}

func load(path string) (*image, error) {
	f, err := elf.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	if f.Machine != elf.EM_X86_64 {
		return nil, fmt.Errorf("unsupported machine %s", f.Machine)
	}
	m := &image{}
	for _, s := range f.Sections {
		if s.Type != elf.SHT_PROGBITS || s.Addr == 0 {
			continue
		}
		d, err := s.Data()
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", s.Name, err)
		}
		m.secs = append(m.secs, s)
		m.data = append(m.data, d)
	}
	if relocs := f.Section(".rela.dyn"); relocs != nil {
		d, err := relocs.Data()
		if err != nil {
			return nil, fmt.Errorf("read .rela.dyn: %w", err)
		}
		for i := 0; i+24 <= len(d); i += 24 {
			if binary.LittleEndian.Uint64(d[i+8:])&0xffffffff != rRelative {
				continue
			}
			if b := m.slice(binary.LittleEndian.Uint64(d[i:]), 8); b != nil {
				copy(b, d[i+16:i+24])
			}
		}
	}
	mod := f.Section(".go.module")
	if mod == nil {
		return nil, errors.New("no .go.module section; not a Go 1.27+ binary")
	}
	if m.slice(mod.Addr, modModuleSize) == nil {
		return nil, errors.New(".go.module is not loaded")
	}
	m.types = m.u64(mod.Addr + modTypes)
	m.typedesclen = m.u64(mod.Addr + modTypedesc)
	return m, nil
}

// slice returns the n bytes at va, or nil if they are not in a loaded section.
func (m *image) slice(va, n uint64) []byte {
	for i, s := range m.secs {
		if va >= s.Addr && va+n <= s.Addr+uint64(len(m.data[i])) {
			o := va - s.Addr
			return m.data[i][o : o+n]
		}
	}
	return nil
}

func (m *image) must(va, n uint64) []byte {
	b := m.slice(va, n)
	if b == nil {
		panic(fmt.Sprintf("va %#x+%d is not mapped", va, n))
	}
	return b
}

func (m *image) u8(va uint64) uint8   { return m.must(va, 1)[0] }
func (m *image) u16(va uint64) uint16 { return binary.LittleEndian.Uint16(m.must(va, 2)) }
func (m *image) u32(va uint64) uint32 { return binary.LittleEndian.Uint32(m.must(va, 4)) }
func (m *image) u64(va uint64) uint64 { return binary.LittleEndian.Uint64(m.must(va, 8)) }

func (m *image) uvarint(va uint64) (uint64, uint64) {
	v, n := binary.Uvarint(m.must(va, min(10, m.avail(va))))
	return v, uint64(n)
}

func (m *image) avail(va uint64) uint64 {
	for i, s := range m.secs {
		if va >= s.Addr && va < s.Addr+uint64(len(m.data[i])) {
			return s.Addr + uint64(len(m.data[i])) - va
		}
	}
	return 0
}

// name decodes an abi.Name.
func (m *image) name(va uint64) (name, tag string, embedded bool) {
	flags := m.u8(va)
	l, n := m.uvarint(va + 1)
	p := va + 1 + n
	name = string(m.must(p, l))
	if flags&(1<<1) != 0 {
		tl, tn := m.uvarint(p + l)
		tag = string(m.must(p+l+tn, tl))
	}
	return name, tag, flags&(1<<3) != 0
}

func (m *image) typeAt(va uint64) rtype {
	t := rtype{va: va, tflag: m.u8(va + 20), kind: m.u8(va + 23)}
	t.str, _, _ = m.name(m.types + uint64(m.u32(va+40)))
	if t.tflag&tflagExtraStar != 0 {
		t.str = strings.TrimPrefix(t.str, "*")
	}
	return t
}

// size mirrors abi.Type.DescriptorSize.
func (m *image) size(t rtype) (uint64, error) {
	var base, add uint64
	switch t.kind {
	case kindArray:
		base = sizeArray
	case kindChan:
		base = sizeChan
	case kindFunc:
		in := uint64(m.u16(t.va + sizeType))
		out := uint64(m.u16(t.va+sizeType+2) & (1<<15 - 1))
		base, add = sizeFunc, (in+out)*8
	case kindInterface:
		base, add = sizeInterface, m.u64(t.va+sizeType+16)*sizeImethod
	case kindMap:
		base = sizeMap
	case kindPointer:
		base = sizePtr
	case kindSlice:
		base = sizeSlice
	case kindStruct:
		base, add = sizeStruct, m.u64(t.va+sizeType+16)*sizeField
	case kindString, kindUnsafePointer:
		base = sizeType
	default:
		if t.kind == 0 || t.kind > kindComplex128 {
			return 0, fmt.Errorf("invalid kind %d at %#x", t.kind, t.va)
		}
		base = sizeType
	}
	if t.tflag&tflagUncommon == 0 {
		return base + add, nil
	}
	return base + sizeUncommon + add + uint64(m.u16(t.va+base+4))*sizeMethod, nil
}

// walk returns the type descriptors in [types+8, types+typedesclen).
func (m *image) walk() ([]rtype, error) {
	end := m.types + m.typedesclen
	var all []rtype
	for td := m.types + 8; td < end; {
		td = (td + 7) &^ 7
		t := m.typeAt(td)
		n, err := m.size(t)
		if err != nil {
			return nil, fmt.Errorf("after %d types: %w", len(all), err)
		}
		all = append(all, t)
		td += n
	}
	return all, nil
}

// structs returns the named structs matching re, reachable from the walked
// descriptors through element and field types, sorted by name.
func (m *image) structs(all []rtype, re *regexp.Regexp) []rtype {
	seen := map[uint64]struct{}{}
	var out []rtype
	var visit func(t rtype)
	visit = func(t rtype) {
		if _, ok := seen[t.va]; ok {
			return
		}
		seen[t.va] = struct{}{}
		switch t.kind {
		case kindArray, kindChan, kindPointer, kindSlice:
			visit(m.typeAt(m.u64(t.va + sizeType)))
		case kindMap:
			visit(m.typeAt(m.u64(t.va + sizeType)))
			visit(m.typeAt(m.u64(t.va + sizeType + 8)))
		case kindStruct:
			if t.tflag&tflagNamed == 0 || !re.MatchString(t.str) {
				return
			}
			out = append(out, t)
			fields := m.u64(t.va + sizeType + 8)
			for i := range m.u64(t.va + sizeType + 16) {
				visit(m.typeAt(m.u64(fields + i*sizeField + 8)))
			}
		default:
		}
	}
	for _, t := range all {
		visit(t)
	}
	slices.SortFunc(out, func(a, b rtype) int { return strings.Compare(a.str, b.str) })
	return out
}

func (m *image) print(t rtype) {
	fmt.Printf("type %s struct {\n", t.str)
	fields := m.u64(t.va + sizeType + 8)
	for i := range m.u64(t.va + sizeType + 16) {
		fv := fields + i*sizeField
		name, tag, embedded := m.name(m.u64(fv))
		typ := m.typeAt(m.u64(fv + 8)).str
		line := "\t" + name + " " + typ
		if embedded {
			line = "\t" + typ
		}
		if tag != "" {
			line += " `" + tag + "`"
		}
		fmt.Println(line)
	}
	fmt.Println("}")
}
